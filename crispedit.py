import random
from copy import deepcopy
from typing import Any, Dict, List
import torch
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
from tools import *
from utils import chunks, save_model_and_tokenizer

from easyeditor.mymodels.hparams import CrispLoRAHyperParams
from easyeditor.models.crispedit.CrispEdit_hparams import CrispEditHyperParams
from easyeditor.models.crispedit.utils import (
    cache_weights_to_cpu, 
    calculate_cov_cache_with_old_data, 
    calculate_cov_cache_with_request, 
    build_optimizer_with_cov_caches, 
    recalculate_cov_cache_if_weights_changed, 
    combine_layer_to_cov_caches, 
    calculate_old_loss, 
    get_weights, 
    calculate_old_edit_loss, 
    wrap_model_with_lora_and_return_opt,
)

from easyeditor.mymodels.scheme_a_projected_lora import (
    wrap_model_and_build_projected_optimizer,
)


def execute_ft(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: CrispEditHyperParams,
    **kwargs: Any,
) -> AutoModelForCausalLM:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    """
    device = model.device
    tracker = kwargs.get("tracker", None)
    if tok.padding_side != "right":
        tok.padding_side = "right"
    
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"] and request["target_new"][0] != " ":
            requests[i]["target_new"] = " " + request["target_new"]
    
    layer_to_cov_cache_old = calculate_cov_cache_with_old_data(
        model, tok, hparams, force_recompute=False
    )
    
    if hparams.perform_lora:
        model, opt = wrap_model_with_lora_and_return_opt(model, hparams)
        current_weights_cpu = None #my code gets uglier with each day
    else:
        opt = build_optimizer_with_cov_caches(model, hparams, [layer_to_cov_cache_old])
        weights = get_weights(model, hparams, bias=True)
        current_weights_cpu = cache_weights_to_cpu(weights)
        for name, w in model.named_parameters():
            w.requires_grad = name in weights
    
    old_loss = calculate_old_loss(model, tok, hparams)
    tracker.log(old_loss) # fine to log even if empty, basically no-op
    
    loss_meter = AverageMeter()
    pbar = trange(hparams.num_steps)

    for it in pbar:
        loss_meter.reset()

        random.shuffle(requests)
        texts = [r["prompt"] for r in requests]
        targets = [r["target_new"] for r in requests]

        # split into batches
        for txt, tgt in zip(
            chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)
        ):
            inputs_targets = [txt_ + tgt_ for txt_, tgt_ in zip(txt, tgt)]
            encodings = tok(inputs_targets, return_tensors="pt", padding=True, truncation=True, max_length=hparams.max_length).to(device)

            labels = encodings["input_ids"].clone()

            labels[labels == tok.pad_token_id] = -100
            for i, prompt in enumerate(txt):
                prompt_len = len(tok(prompt, add_special_tokens=True, truncation=True, max_length=hparams.max_length)["input_ids"])
                labels[i, :prompt_len] = -100
            opt.zero_grad(set_to_none=True)
            outputs = model(**encodings, labels=labels)
            loss = outputs.loss
                
            loss_meter.update(loss.item(), n=labels.size(0))
            
            if loss.item() >= 1e-2:
                loss.backward()
                opt.step()
                current_weights_cpu, layer_to_cov_cache_old, should_recalculate = recalculate_cov_cache_if_weights_changed(
                    model,
                    tok,
                    hparams,
                    current_weights_cpu,
                    layer_to_cov_cache_old,
                )
                if should_recalculate:
                    opt = build_optimizer_with_cov_caches(model, hparams, [layer_to_cov_cache_old], opt=opt)

        metrics = calculate_old_loss(model, tok, hparams)
        metrics.update({f"FT Loss": loss_meter.avg})
        tracker.log(metrics) # fine to log even if empty, basically no-op
        
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})
        if loss_meter.avg < 1e-2:
            break
    
    if hparams.perform_lora:
        model = model.merge_and_unload()
    
    return model

def execute_ft_sequential(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: CrispEditHyperParams,
    **kwargs: Any,
) -> AutoModelForCausalLM:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    """
    tracker = kwargs.get("tracker", None)
    device = model.device
    
    if tok.padding_side != "right":
        tok.padding_side = "right"
    
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"] and request["target_new"][0] != " ":
            requests[i]["target_new"] = " " + request["target_new"]
    random.shuffle(requests)
    texts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]
    txt_chunks, tgt_chunks = [], []


    layer_to_cov_cache_old = calculate_cov_cache_with_old_data(
        model, tok, hparams, force_recompute=False
    )

    
    if hparams.perform_lora:
        model, opt = wrap_model_with_lora_and_return_opt(model, hparams)
        current_weights_cpu = None #my code gets uglier with each day
    else:
        opt = build_optimizer_with_cov_caches(model, hparams, [layer_to_cov_cache_old])
        weights = get_weights(model, hparams, bias=True)
        current_weights_cpu = cache_weights_to_cpu(weights)
        
        for name, w in model.named_parameters():
            w.requires_grad = name in weights

    old_loss = calculate_old_loss(model, tok, hparams)
    tracker.log(old_loss) # fine to log even if empty, basically no-op
    
    layer_to_cov_cache_data = None
    loss_meter = AverageMeter()

    # split into batches
    for txt_edit, tgt_edit in zip(
        chunks(texts, hparams.num_edits), chunks(targets, hparams.num_edits)
    ):
        pbar = trange(hparams.num_steps)
        for it in pbar:
            loss_meter.reset()
            for txt, tgt in zip(
                chunks(txt_edit, hparams.batch_size), chunks(tgt_edit, hparams.batch_size)
            ):
                inputs_targets = [txt_ + tgt_ for txt_, tgt_ in zip(txt, tgt)]
                encodings = tok(inputs_targets, return_tensors="pt", padding=True, truncation=True, max_length=hparams.max_length).to(device)

                labels = encodings["input_ids"].clone()

                labels[labels == tok.pad_token_id] = -100
                for i, prompt in enumerate(txt):
                    prompt_len = len(tok(prompt, add_special_tokens=True, truncation=True, max_length=hparams.max_length)["input_ids"])
                    labels[i, :prompt_len] = -100
                opt.zero_grad()
                outputs = model(**encodings, labels=labels)
                loss = outputs.loss

                if loss.item() >= 1e-2:
                    loss.backward()
                    opt.step()
                    current_weights_cpu, layer_to_cov_cache_old, should_recalculate = recalculate_cov_cache_if_weights_changed(
                        model,
                        tok,
                        hparams,
                        current_weights_cpu,
                        layer_to_cov_cache_old,
                    )
                    if should_recalculate:                            
                        if hparams.edit_n_samples > 0 and len(txt_chunks) > 0:
                            old_txt_list = [item for sublist in txt_chunks for item in sublist]
                            old_tgt_list = [item for sublist in tgt_chunks for item in sublist]

                            layer_to_cov_cache_data = calculate_cov_cache_with_request(
                                old_txt_list,
                                old_tgt_list,
                                model,
                                tok,
                                hparams,
                            )
                            opt = build_optimizer_with_cov_caches(model, hparams, [layer_to_cov_cache_old, layer_to_cov_cache_data], opt=opt)
                        else:
                            opt = build_optimizer_with_cov_caches(model, hparams, [layer_to_cov_cache_old] if layer_to_cov_cache_data is None else [layer_to_cov_cache_old, layer_to_cov_cache_data], opt=opt)
                loss_meter.update(loss.item(), n=labels.size(0))
            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})
            if loss_meter.avg < 1e-2:
                break
        print(f"Loss after editing number of samples {len(txt_edit)}: {loss_meter.avg}")
        
        txt_chunks.append(txt_edit)
        tgt_chunks.append(tgt_edit)
        
        if hparams.edit_cache_style == 'sequential':
            layer_to_cov_cache_data_new = calculate_cov_cache_with_request(
                txt_edit,
                tgt_edit,
                model,
                tok,
                hparams,
            )
            if layer_to_cov_cache_data is None:
                layer_to_cov_cache_data = layer_to_cov_cache_data_new
            else:
                layer_to_cov_cache_data = combine_layer_to_cov_caches([layer_to_cov_cache_data, layer_to_cov_cache_data_new], normalize_trace_with_first=True)
            opt = build_optimizer_with_cov_caches(model, hparams, [layer_to_cov_cache_old, layer_to_cov_cache_data], opt=opt)

        elif hparams.edit_cache_style == 'mix':
            old_txt_list = [item for sublist in txt_chunks for item in sublist]
            old_tgt_list = [item for sublist in tgt_chunks for item in sublist]

            layer_to_cov_cache_data_pretrain_mix = calculate_cov_cache_with_request(
                old_txt_list,
                old_tgt_list,
                model,
                tok,
                hparams,
            )

            opt = build_optimizer_with_cov_caches(model, hparams, [layer_to_cov_cache_data_pretrain_mix], opt=opt)

        metrics = calculate_old_loss(model, tok, hparams)
        old_edit_loss = calculate_old_edit_loss(txt_chunks, tgt_chunks, model, tok)
        metrics.update(old_edit_loss)
        tracker.log(metrics) # fine to log even if empty, basically no-op

    if hparams.perform_lora:
        model = model.merge_and_unload()
    return model

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def setup_requests_for_safeedit(requests: List[Dict]) -> List[Dict]:
    # just a simple way to make safeedit dataset work with old crispedit code
    if "target_new" in requests[0]:
        return requests

    new_requests = []
    for request in requests:
        new_request = {
            'prompt': request['question'],
            'target_new': request['target_unsafe'],
        }
        new_requests.append(new_request)
    return new_requests

def inspect_model_structure(model):
    """
    Thoroughly inspects a Hugging Face/PyTorch model to debug 
    trainable parameters, frozen weights, and buffers.
    """
    print("=" * 100)
    print(f"MODEL DEBUG INSPECTION")
    print("=" * 100)
    print(f"Model Class: {type(model).__name__}")
    print(f"Model Mode:  {'TRAINING' if model.training else 'EVAL'} (affects Dropout/BatchNorm)")
    print("-" * 100)

    total_params = 0
    trainable_params = 0
    frozen_params = 0
    buffer_count = 0
    total_memory_bytes = 0
    
    # --- 1. PARAMETERS (Weights & Biases) ---
    print(f"{'PARAMETER NAME':<55} | {'SHAPE':<20} | {'DTYPE':<10} | {'TRAINABLE'}")
    print("-" * 100)
    
    for name, param in model.named_parameters():
        # Stats
        num_params = param.numel()
        mem_size = num_params * param.element_size()
        total_memory_bytes += mem_size
        total_params += num_params
        
        # Trainable status
        if param.requires_grad:
            trainable_params += num_params
            grad_status = "✅ YES"
        else:
            frozen_params += num_params
            grad_status = "🔒 NO"
            
        # Print row
        print(f"{name:<55} | {str(tuple(param.shape)):<20} | {str(param.dtype).replace('torch.', ''):<10} | {grad_status}")

    print("-" * 100)
    
    # --- 2. BUFFERS (Non-trainable states like BN running means, position IDs) ---
    # These are often overlooked but are "changable" during forward pass!
    buffers = list(model.named_buffers())
    if buffers:
        print("\n" + "=" * 100)
        print("BUFFERS (Non-trainable state, e.g., Running Mean/Var, Position IDs)")
        print("-" * 100)
        print(f"{'BUFFER NAME':<55} | {'SHAPE':<20} | {'DTYPE':<10}")
        print("-" * 100)
        for name, buf in buffers:
            buffer_count += 1
            mem_size = buf.numel() * buf.element_size()
            total_memory_bytes += mem_size
            print(f"{name:<55} | {str(tuple(buf.shape)):<20} | {str(buf.dtype).replace('torch.', ''):<10}")
    
    # --- 3. SUMMARY STATS ---
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("-" * 100)
    print(f"Total Parameters:    {total_params:,}")
    print(f"Trainable Params:    {trainable_params:,} ({100 * trainable_params / total_params if total_params > 0 else 0:.2f}%)")
    print(f"Frozen Params:       {frozen_params:,}")
    print(f"Total Buffers:       {buffer_count}")
    print(f"Approx Model Size:   {total_memory_bytes / (1024**2):.2f} MB")
    print("=" * 100)


def execute_ft_mylora(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: CrispLoRAHyperParams,
    **kwargs: Any,
) -> AutoModelForCausalLM:
    """
    使用方案A执行知识编辑：LoRA + KFac 边缘化投影 Adam 优化器。

    与 execute_ft 的区别：
      - 用 peft 库为模型添加 LoRA 适配器，原始权重全程冻结
      - 将标准 Adam 替换为 ProjectedLoRAOptimizer：
          lora_A 的梯度通过右投影（Ua / mask_a）投影到低能量安全子空间
          lora_B 的梯度通过左投影（Ub / mask_b）投影到低能量安全子空间
        从而保证每步更新只发生在对旧知识影响最小的方向上
      - 训练完成后将 LoRA 权重合并回原始模型，返回普通模型
    """
    tracker = kwargs.get("tracker", None)
    device = model.device

    if tok.padding_side != "right":
        tok.padding_side = "right"

    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"] and request["target_new"][0] != " ":
            requests[i]["target_new"] = " " + request["target_new"]


    # 方法一核心
    peft_model, opt, layer_to_proj_cache = wrap_model_and_build_projected_optimizer(
        model, tok, hparams
    )

    # SchemeA 模式：跳过训练前的旧知识损失计算
    # calculate_old_loss 每次需要在 Wikipedia 上跑 100 条前向传播（约 4~5 分钟），
    # SchemeA 通过 KFac 投影在优化方向上已保证对旧知识影响最小，无需每步监控。
    if not getattr(hparams, "disable_old_loss_check", False):
        old_loss = calculate_old_loss(peft_model, tok, hparams)
        tracker.log(old_loss)

    loss_meter = AverageMeter()
    pbar = trange(hparams.num_steps)

    for it in pbar:
        loss_meter.reset()

        random.shuffle(requests)
        texts = [r["prompt"] for r in requests]
        targets = [r["target_new"] for r in requests]

        for txt, tgt in zip(
            chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)
        ):
            inputs_targets = [txt_ + tgt_ for txt_, tgt_ in zip(txt, tgt)]
            encodings = tok(inputs_targets, return_tensors="pt", padding=True, truncation=True, max_length=hparams.max_length).to(device)

            labels = encodings["input_ids"].clone()

            labels[labels == tok.pad_token_id] = -100
            for i, prompt in enumerate(txt):
                prompt_len = len(tok(prompt, add_special_tokens=True, truncation=True, max_length=hparams.max_length)["input_ids"])
                labels[i, :prompt_len] = -100

            opt.zero_grad(set_to_none=True)
            outputs = peft_model(**encodings, labels=labels)
            loss = outputs.loss

            loss_meter.update(loss.item(), n=labels.size(0))

            if loss.item() >= 1e-2:
                loss.backward()
                opt.step()

        # SchemeA 模式：跳过每步的旧知识损失计算（是性能瓶颈所在）
        # calculate_old_loss 在 disable_old_loss_check=True 时会立即返回 {}，
        # 但如果字段未同步到服务器，仍会执行完整的 Wikipedia 前向传播（4~5 分钟/步）。
        # 因此这里用 getattr 做防御性检查，彻底规避。
        if not getattr(hparams, "disable_old_loss_check", False):
            metrics = calculate_old_loss(peft_model, tok, hparams)
            metrics.update({"FT LoRA Loss": loss_meter.avg})
            tracker.log(metrics)
        else:
            tracker.log({"FT LoRA Loss": loss_meter.avg})

        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})
        if loss_meter.avg < 1e-2:
            break

    # 将 LoRA 适配器权重合并回原始模型权重，
    # 移除适配器结构，返回标准的 AutoModelForCausalLM。
    merged_model = peft_model.merge_and_unload()
    return merged_model