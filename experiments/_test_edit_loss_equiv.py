#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数值一致性自检：build_edit_batches + edit_loss  vs  calculate_request_loss 的逐字逻辑。

不依赖 HF 模型/网络：用一个确定性的微型模型 + mock tokenizer，把两条数据通路
（mask 构造、shift、cross_entropy sum/ignore_index、归一化）跑在同一组输入上，
断言两者逐 token 等价。

run:
    D:/Anaconda3/envs/DL/python.exe experiments/_test_edit_loss_equiv.py
"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


# ---- verbatim copies from run_experiment1_rho_vs_hc.py (data path only) ----
def _chunks(arr, n):
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def dict_to_(data, device):
    for k in data:
        data[k] = data[k].to(device)
    return data


def build_edit_batches(tok, edit_txt, edit_tgt, batch_size=32):
    pad_id = tok.pad_token_id
    fill_id = pad_id if pad_id is not None else 0
    encoded = []
    for t, g in zip(edit_txt, edit_tgt):
        ids = tok(t + g, return_tensors=None, add_special_tokens=True)["input_ids"]
        prompt_len = len(tok.encode(t, add_special_tokens=True))
        encoded.append((ids, prompt_len))
    batches = []
    for chunk in _chunks(encoded, batch_size):
        max_len = max(len(ids) for ids, _ in chunk)
        input_ids = torch.full((len(chunk), max_len), fill_id, dtype=torch.long)
        attention_mask = torch.zeros((len(chunk), max_len), dtype=torch.long)
        labels = torch.full((len(chunk), max_len), -100, dtype=torch.long)
        for i, (ids, prompt_len) in enumerate(chunk):
            L = len(ids)
            ids_t = torch.tensor(ids, dtype=torch.long)
            input_ids[i, :L] = ids_t
            attention_mask[i, :L] = 1
            labels[i, :L] = ids_t
        if pad_id is not None:
            labels[labels == pad_id] = -100
        for i, (_, prompt_len) in enumerate(chunk):
            labels[i, :prompt_len] = -100
        batches.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})
    return batches


def edit_loss(model, edit_batches):
    model.eval()
    total_loss = torch.zeros((), device=model.device, dtype=torch.float32)
    total_tokens = torch.zeros((), device=model.device, dtype=torch.float32)
    with torch.no_grad():
        for batch in edit_batches:
            batch = dict_to_(batch, model.device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_loss += loss
            total_tokens += (shift_labels != -100).sum()
    tt = total_tokens.item()
    if tt == 0:
        return 0.0
    return total_loss.item() / tt


# ---- verbatim re-implementation of calculate_request_loss (batch=1, float64 acc) ----
def ref_request_loss(model, tok, txt, tgt, sample_size=1):
    """逐字复刻 layer_stats.calculate_request_loss 的数值逻辑（batch_size=1）。"""
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    batch_size = 1
    with torch.no_grad():
        for txt_edit, tgt_edit in zip(_chunks(txt, batch_size), _chunks(tgt, batch_size)):
            inputs_targets = [t + g for t, g in zip(txt_edit, tgt_edit)]
            encodings = tok(inputs_targets, return_tensors="pt", padding=True)
            encodings = {k: v.to(model.device) for k, v in encodings.items()}
            labels = encodings["input_ids"].clone()
            for i, prompt in enumerate(txt_edit):
                prompt_len = len(tok.encode(prompt, add_special_tokens=True))
                labels[i, :prompt_len] = -100
            if tok.pad_token_id is not None:
                labels[labels == tok.pad_token_id] = -100
            outputs = model(**encodings)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            current_valid_tokens = (shift_labels != -100).sum().item()
            total_loss += loss.item()
            total_tokens += current_valid_tokens
    return total_loss / total_tokens if total_tokens > 0 else 0.0


# ---- mock tokenizer: char-level, deterministic, with pad token ----
class MockTok:
    """字符级 mock tokenizer，满足两条通路用到的接口。"""
    def __init__(self):
        self.vocab = {ch: i for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz 0.,'")}
        # 留一个特殊 pad id（不在常规字符集）
        self._pad_id = len(self.vocab)
        self._bos = len(self.vocab) + 1
        self._eos = len(self.vocab) + 2
        self.vocab["[PAD]"] = self._pad_id
        self.vocab["[BOS]"] = self._bos
        self.vocab["[EOS]"] = self._eos
        self.pad_token = "[PAD]"
        self.pad_token_id = self._pad_id
        self.bos_token_id = self._bos
        self.eos_token_id = self._eos
        self.padding_side = "right"  # calculate_request_loss 的隐含假设

    def _encode(self, text, add_special_tokens):
        ids = []
        if add_special_tokens:
            ids.append(self._bos)
        for ch in text:
            if ch in self.vocab:
                ids.append(self.vocab[ch])
        if add_special_tokens:
            ids.append(self._eos)
        return ids

    def encode(self, text, add_special_tokens=True):
        return self._encode(text, add_special_tokens)

    def __call__(self, texts, return_tensors=None, padding=False, add_special_tokens=True):
        if isinstance(texts, str):
            texts = [texts]
        seqs = [self._encode(t, add_special_tokens) for t in texts]
        if padding:
            max_len = max(len(s) for s in seqs)
            fill = self._pad_id
            padded = [s + [fill] * (max_len - len(s)) for s in seqs]  # right pad
            attn = [[1] * len(s) + [0] * (max_len - len(s)) for s in seqs]
        else:
            padded = seqs
            attn = [[1] * len(s) for s in seqs]
        out = {
            "input_ids": padded,
            "attention_mask": attn,
        }
        if return_tensors == "pt":
            out = {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}
        return out


# ---- tiny deterministic model ----
class OutObj:
    def __init__(self, logits):
        self.logits = logits


class TinyModel(nn.Module):
    """确定性的微型因果 LM：embed -> 一层线性 -> unembed。"""
    def __init__(self, vocab_size, d=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.linear = nn.Linear(d, d)
        self.unembed = nn.Linear(d, vocab_size, bias=False)
        self.device = torch.device("cpu")
        # 固定种子让权重确定
        with torch.no_grad():
            g = torch.Generator().manual_seed(123)
            self.embed.weight.copy_(torch.randn(vocab_size, d, generator=g) * 0.1)
            self.linear.weight.copy_(torch.randn(d, d, generator=g) * 0.1)
            self.linear.bias.copy_(torch.randn(d, generator=g) * 0.1)
            self.unembed.weight.copy_(torch.randn(vocab_size, d, generator=g) * 0.1)

    def forward(self, input_ids=None, attention_mask=None, use_cache=False, **kw):
        h = self.embed(input_ids)
        h = self.linear(h)
        logits = self.unembed(h)
        return OutObj(logits)


def main():
    torch.manual_seed(0)
    tok = MockTok()
    model = TinyModel(tok._pad_id + 3, d=16)

    # 构造若干编辑样本：prompt + target_new（注意 target 含空格/标点覆盖多 token）
    edit_txt = [
        "the capital of france",
        "water boils at",
        "the largest planet is",
        "a square has",
        "the speed of light is",
        "shakespeare wrote",
        "the chemical symbol for gold is",
        "the sun rises in the",
    ]
    edit_tgt = [
        " paris",
        " 100 degrees",
        " jupiter",
        " four sides",
        " very fast",
        " many plays",
        " au",
        " east",
    ]

    # 1) batch_size=1 的 edit_loss 应与 ref_request_loss 近乎相等
    batches_b1 = build_edit_batches(tok, edit_txt, edit_tgt, batch_size=1)
    L_b1 = edit_loss(model, batches_b1)
    L_ref = ref_request_loss(model, tok, edit_txt, edit_tgt, sample_size=len(edit_txt))
    print(f"[b=1 ] edit_loss = {L_b1:.10f}")
    print(f"[ref ] ref_loss  = {L_ref:.10f}")
    diff1 = abs(L_b1 - L_ref)
    print(f"[b=1 ] |diff| = {diff1:.3e}")
    assert diff1 < 1e-5, f"batch=1 mismatch: {diff1}"
    print("[b=1 ] OK (numerically consistent with calculate_request_loss)")

    # 2) 不同 batch_size 的 edit_loss 必须彼此一致（批大小不影响结果）
    for bs in [2, 3, 4, 8, 100]:
        batches = build_edit_batches(tok, edit_txt, edit_tgt, batch_size=bs)
        L = edit_loss(model, batches)
        diff = abs(L - L_ref)
        print(f"[b={bs:<3}] edit_loss = {L:.10f}  |diff_vs_ref| = {diff:.3e}")
        assert diff < 1e-5, f"batch={bs} mismatch: {diff}"
    print("[batch] OK (batch_size invariant)")

    # 3) mask 正确性：单条样本的 labels 只有 target_new token 不为 -100
    b1 = build_edit_batches(tok, [edit_txt[0]], [edit_tgt[0]], batch_size=1)[0]
    ids = b1["input_ids"][0]
    labels = b1["labels"][0]
    prompt_len = len(tok.encode(edit_txt[0], add_special_tokens=True))
    full_len = len(tok.encode(edit_txt[0] + edit_tgt[0], add_special_tokens=True))
    # target 区域 = [prompt_len : full_len]，应等于 labels != -100 的位置
    expected_mask_pos = torch.arange(ids.size(0))
    expected = ((expected_mask_pos >= prompt_len) & (expected_mask_pos < full_len))
    got = (labels != -100)
    assert torch.equal(expected, got), f"mask mismatch:\nexpected={expected}\ngot={got}"
    print(f"[mask ] OK (prompt_len={prompt_len}, target_len={full_len}, "
          f"target positions exactly {[i for i in range(full_len) if i>=prompt_len]})")

    # 4) pad 位置确实被 mask（labels 里无 pad_id）
    assert (labels == tok.pad_token_id).sum().item() == 0, "pad token leaked into labels"
    print("[pad  ] OK (pad positions masked to -100)")

    print("\nALL CHECKS PASSED — build_edit_batches/edit_loss 数值与 calculate_request_loss 一致。")


if __name__ == "__main__":
    main()
