import math
import torch
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..hparams import CrispLoRAHyperParams
from ...tools import *

def _compute_loss(
    model,
    tok: AutoTokenizer,
    texts: List[str],
    targets: List[str],
    device: torch.device,
    hparams: CrispLoRAHyperParams,
) -> torch.Tensor:
    inputs_targets = [t + tg for t, tg in zip(texts, targets)]
    encodings = tok(
        inputs_targets,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
    ).to(device)

    labels = encodings["input_ids"].clone()
    labels[labels == tok.pad_token_id] = -100
    for i, prompt in enumerate(texts):
        prompt_len = len(
            tok(prompt, add_special_tokens=True, truncation=True,
                max_length=hparams.max_length)["input_ids"]
        )
        labels[i, :prompt_len] = -100

    return model(**encodings, labels=labels).loss


def _chunks(lst: List, n: int):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def apply_simple_finetune(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: list,
    hparams: CrispLoRAHyperParams,
    **kwargs
):
    if tok.padding_side != "right":
        tok.padding_side = "right"
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 解冻目标层
    for layer_idx in hparams.layers:
        module_name = hparams.rewrite_module_tmp.format(layer_idx)
        # 去掉末尾的weight
        module_name = module_name[:-len(".weight")] if module_name.endswith(".weight") else module_name
        module = model.get_submodule(module_name)
        for param in module.parameters():
            print(f"解冻目标层:{module_name}")
            param.requires_grad = True

    # 优化器
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hparams.lr,
        weight_decay=hparams.weight_decay
    )
    for i, request in enumerate(requests):
        if request["target_new"] and request["target_new"][0] != " ":
            requests[i]["target_new"] = " " + request["target_new"]

    device = next(model.parameters()).device

    texts = [
        r["prompt"].format(r.get("subject", "")) if "{}" in r["prompt"] else r["prompt"]
        for r in requests
    ]
    targets = [r["target_new"] for r in requests]

    for step in range(hparams.num_steps):
        total_loss = 0.0
        for txt_batch, tgt_batch in zip(
            _chunks(texts, hparams.batch_size),
            _chunks(targets, hparams.batch_size),
        ):
            optimizer.zero_grad()
            loss = _compute_loss(model, tok, txt_batch, tgt_batch, device, hparams)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        num_batches = max(1, math.ceil(len(texts) / hparams.batch_size))
        avg_loss = total_loss / num_batches
        ExperimentTracker.log({"LOSS": avg_loss})
        print(f"Step {step+1}/{hparams.num_steps}, Loss: {avg_loss:.4f}")
        if avg_loss < 1e-3:
            print("损失收敛，提前结束训练")
            break
    return model
