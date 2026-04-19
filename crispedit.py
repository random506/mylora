import random
from copy import deepcopy
from typing import Any, Dict, List
import torch
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
from tools import *
from utils import chunks, save_model_and_tokenizer

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

from easyeditor.mymodels import (
    CrispLoRAHyperParams,
    wrap_model_and_build_projected_optimizer,
    build_lora_projection_cache,
    attach_curvature_lora_variant,
    set_curvature_bases,
)
from peft import LoraConfig, get_peft_model, TaskType


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


def execute_ft_grad_lora(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: CrispLoRAHyperParams,
    **kwargs: Any,
) -> AutoModelForCausalLM:
    # 对于梯度进行投影
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


def execute_ft_param_lora(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: CrispLoRAHyperParams,
    **kwargs: Any,
) -> AutoModelForCausalLM:
 # 对 LoRA 参数本身做曲率约束（CurvatureLora），使增量落在低曲率子空间
    tracker = kwargs.get("tracker", None)
    device = model.device

    if tok.padding_side != "right":
        tok.padding_side = "right"

    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"] and request["target_new"][0] != " ":
            requests[i]["target_new"] = " " + request["target_new"]

    # Step 1: 计算 K-FAC 协方差  特征分解
    layer_to_cov_cache_old = build_lora_projection_cache(model,tok,hparams)

    # Step 2: 挂载标准 LoRA
    model.config.use_cache = False
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=hparams.lora_rank,
        lora_alpha=hparams.lora_alpha,
        lora_dropout=hparams.lora_dropout,
        layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,
        target_modules=hparams.target_modules,
    )

    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    adapter_name = "default"

    # Step 3: 将标准 LoRA 层后挂载为 CurvatureLora variant
    attach_curvature_lora_variant(peft_model, adapter_name=adapter_name)

    # Step 4: 从 K-FAC 特征向量中提取高曲率方向基，写入每个 LoRA 层
    # 高曲率方向 = 协方差特征值大的方向（eigh 升序，高能量列 mask_a==False 即为高曲率）
    for name, module in peft_model.named_modules():
        if not (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and hasattr(module, "in_features")
            and hasattr(module, "out_features")
        ):
            continue
        if adapter_name not in getattr(module, "lora_variant", {}):
            continue

        matched_cache = None
        for layer_name, cache in layer_to_cov_cache_old.items():
            clean = layer_name[: -len(".weight")] if layer_name.endswith(".weight") else layer_name
            if clean in name:
                matched_cache = cache
                break

        if matched_cache is None:
            continue

        base_weight = module.base_layer.weight if hasattr(module, "base_layer") else module.weight
        dev = base_weight.device
        dtype = base_weight.dtype

        # Ua 列按特征值升序排列；高曲率方向 = 高能量方向 = mask_a==False 的列
        Ua = matched_cache["Ua"].to(device=dev, dtype=dtype)  # (d_in, d_in)
        Ub = matched_cache["Ub"].to(device=dev, dtype=dtype)  # (d_out, d_out)
        mask_a = matched_cache["mask_a"].to(device=dev)  # True = 低能量(安全)
        mask_b = matched_cache["mask_b"].to(device=dev)

        # 高曲率方向 = ~mask（不安全方向）
        U_in_bar = Ua[:, ~mask_a]   # (d_in, k_in)
        U_out_bar = Ub[:, ~mask_b]  # (d_out, k_out)

        setattr(module, f"U_in_bar_{adapter_name}", U_in_bar)
        setattr(module, f"U_out_bar_{adapter_name}", U_out_bar)

    # Step 5: 构建普通 Adam 优化器（CurvatureLora 通过 forward 结构约束增量，无需特殊优化器）
    opt = torch.optim.Adam(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )

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
            encodings = tok(
                inputs_targets,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=hparams.max_length,
            ).to(device)

            labels = encodings["input_ids"].clone()
            labels[labels == tok.pad_token_id] = -100
            for i, prompt in enumerate(txt):
                prompt_len = len(
                    tok(
                        prompt,
                        add_special_tokens=True,
                        truncation=True,
                        max_length=hparams.max_length,
                    )["input_ids"]
                )
                labels[i, :prompt_len] = -100

            opt.zero_grad(set_to_none=True)
            outputs = peft_model(**encodings, labels=labels)
            loss = outputs.loss

            loss_meter.update(loss.item(), n=labels.size(0))

            if loss.item() >= 1e-2:
                loss.backward()
                opt.step()

        if not getattr(hparams, "disable_old_loss_check", False):
            metrics = calculate_old_loss(peft_model, tok, hparams)
            metrics.update({"FT ParamLoRA Loss": loss_meter.avg})
            tracker.log(metrics)
        else:
            tracker.log({"FT ParamLoRA Loss": loss_meter.avg})

        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})
        if loss_meter.avg < 1e-2:
            break

    merged_model = peft_model.merge_and_unload()
    return merged_model
    
    