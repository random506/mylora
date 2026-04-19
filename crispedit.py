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
    """
    基于 K-FAC 曲率感知的 LoRA 微调（CurvatureLora / ParamLora 方法）。

    核心思想：
        标准 LoRA 的参数更新 ΔW = B·A 可以在权重空间的任意方向移动，
        容易损坏与已有知识相关的高曲率方向（Fisher 信息大的方向）。
        本方法在挂载 LoRA 之前，先利用 K-FAC 对预训练模型的曲率进行
        分析，得到每一层输入/输出的特征分解（Kronecker 因子的特征向量）。
        随后通过 CurvatureLora variant 在 forward 过程中将 LoRA 增量
        正交投影到"低曲率子空间"，从而在学习新知识的同时保护旧知识。

    与 execute_ft_grad_lora 的区别：
        - execute_ft_grad_lora：在梯度上做投影（梯度空间约束）。
        - execute_ft_param_lora：在参数结构上做约束（forward 时增量被
          限定在安全子空间内，无需修改优化器）。

    数据流总览：
    ┌──────────────────────────────────────────────────────────────────┐
    │  输入                                                            │
    │  model: 预训练 CausalLM（冻结基础权重）                          │
    │  tok:   对应的 Tokenizer                                         │
    │  requests: [{"prompt": str, "target_new": str}, ...]            │
    │  hparams: LoRA rank/alpha/dropout、层选择、lr 等超参              │
    └────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  Step 1: K-FAC 协方差缓存构建                                     │
    │  build_lora_projection_cache(model, tok, hparams)               │
    │    → 对目标层做一次前向，收集每层的 Kronecker 因子 (A_cov, B_cov)  │
    │    → 对因子做特征分解，得到特征向量矩阵 Ua(d_in×d_in)、           │
    │      Ub(d_out×d_out) 以及安全/危险方向掩码 mask_a、mask_b         │
    │  输出: layer_to_cov_cache_old                                    │
    │    { "layer.weight": {"Ua": Tensor, "Ub": Tensor,               │
    │                        "mask_a": BoolTensor, "mask_b": BoolTensor} }│
    └────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  Step 2: 挂载标准 LoRA（PEFT）                                   │
    │  model.config.use_cache = False   # 关闭 KV cache 以支持梯度     │
    │  model.enable_input_require_grads()                              │
    │  LoraConfig(r, alpha, dropout, layers, target_modules)          │
    │  peft_model = get_peft_model(model, peft_config)                │
    │    → 在目标线性层外包裹 LoRA 结构:                                │
    │      原始 W(d_out×d_in) 保持冻结                                 │
    │      新增可训练: lora_A(r×d_in)，lora_B(d_out×r)                │
    │      输出 = W·x + (lora_alpha/r) * lora_B·lora_A·x             │
    └────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  Step 3: 将 LoRA 层升级为 CurvatureLora variant                  │
    │  attach_curvature_lora_variant(peft_model, adapter_name)        │
    │    → 在每个 LoRA 模块的 lora_variant 字典中注入新的 forward 钩子  │
    │    → 新 forward: 增量 ΔW = B·A 先通过 U_out_bar / U_in_bar       │
    │      做正交投影，确保更新落在"高曲率（危险）子空间"的正交补（      │
    │      即低曲率安全子空间）                                         │
    └────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  Step 4: 将曲率基向量注入每个 CurvatureLora 模块                  │
    │  遍历 peft_model.named_modules():                                │
    │    - 筛选同时具有 lora_A、lora_B、in_features、out_features 的层  │
    │    - 在 layer_to_cov_cache_old 中匹配对应层名                    │
    │    - 提取高曲率方向（~mask）:                                     │
    │        U_in_bar  = Ua[:, ~mask_a]  # (d_in,  k_in)  输入高曲率基│
    │        U_out_bar = Ub[:, ~mask_b]  # (d_out, k_out) 输出高曲率基│
    │    - setattr 注入模块，供 CurvatureLora forward 使用              │
    │  几何含义:                                                        │
    │    mask==True  → 低能量（低曲率）方向 = 安全方向（可修改）         │
    │    mask==False → 高能量（高曲率）方向 = 危险方向（应避开）         │
    │    投影到危险方向的正交补 ≡ 只在安全子空间内更新权重               │
    └────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  Step 5: 构建 Adam 优化器                                         │
    │  只优化 requires_grad=True 的参数（即 lora_A、lora_B）            │
    │  不需要特殊优化器——子空间约束已在 CurvatureLora forward 中实现     │
    └────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  Step 6: 训练循环（num_steps 轮）                                 │
    │  每轮:                                                            │
    │    ① 随机打乱 requests                                            │
    │    ② 按 batch_size 切分，拼接 prompt+target_new 作为输入          │
    │       tokenize → input_ids, attention_mask                       │
    │    ③ 构建 labels:                                                 │
    │       - pad token 位置置 -100（不计算 loss）                      │
    │       - prompt 长度内的 token 置 -100（只对 target 计算 loss）    │
    │    ④ 前向: peft_model(**encodings, labels=labels)                │
    │       → CurvatureLora forward 将增量投影到安全子空间              │
    │       → 计算 CrossEntropy loss（仅在 target 位置）                │
    │    ⑤ 若 loss >= 1e-2: 反向传播 + Adam step                       │
    │    ⑥ 记录 metrics（旧数据 loss + 当前 FT ParamLoRA Loss）         │
    │    ⑦ 若 loss_meter.avg < 1e-2 则提前停止                         │
    └────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  Step 7: LoRA 权重合并与返回                                      │
    │  peft_model.merge_and_unload()                                  │
    │    → 将 (lora_alpha/r)*lora_B·lora_A 吸收回原始权重 W            │
    │    → 移除 PEFT 适配器结构，返回标准 AutoModelForCausalLM          │
    └──────────────────────────────────────────────────────────────────┘

    Args:
        model:    预训练的因果语言模型（权重将在原地被 LoRA 适配器修改）。
        tok:      与 model 对应的分词器。
        requests: 编辑请求列表，每项包含:
                    - "prompt"     (str): 输入提示文本
                    - "target_new" (str): 期望的新输出文本
        hparams:  CrispLoRAHyperParams，包含:
                    lora_rank, lora_alpha, lora_dropout,
                    layers, target_modules, lr, weight_decay,
                    batch_size, num_steps, max_length,
                    disable_old_loss_check 等。
        **kwargs: 可选关键字参数，目前支持:
                    - tracker: 用于记录训练指标的 tracker 对象。

    Returns:
        merged_model (AutoModelForCausalLM):
            已将 CurvatureLora 增量合并回基础权重的模型，
            与输入 model 结构相同，无 PEFT 适配器层。
    """
    tracker = kwargs.get("tracker", None)
    device = model.device

    # 确保 padding 在右侧，与自回归 loss 掩码逻辑一致
    if tok.padding_side != "right":
        tok.padding_side = "right"

    # 深拷贝请求列表，避免修改原始数据；
    # 确保 target_new 以空格开头（与 tokenizer 的 BOS 处理方式对齐）
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"] and request["target_new"][0] != " ":
            requests[i]["target_new"] = " " + request["target_new"]

    # ------------------------------------------------------------------ #
    # Step 1: 计算 K-FAC 协方差缓存并做特征分解
    #   数据流: model + tok + hparams
    #        → 对目标层前向一次，收集激活统计量
    #        → 构建 Kronecker 因子并分解为 (Ua, Ub, mask_a, mask_b)
    #        → 返回 {layer_name: {"Ua", "Ub", "mask_a", "mask_b"}}
    # ------------------------------------------------------------------ #
    layer_to_cov_cache_old = build_lora_projection_cache(model, tok, hparams)

    # ------------------------------------------------------------------ #
    # Step 2: 挂载标准 LoRA 适配器
    #   数据流: model → peft_model（基础权重冻结，新增 lora_A/lora_B 可训练）
    # ------------------------------------------------------------------ #
    model.config.use_cache = False       # 关闭 KV cache，允许梯度回传
    model.enable_input_require_grads()   # 确保输入 embedding 支持梯度

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=hparams.lora_rank,                                              # LoRA 秩
        lora_alpha=hparams.lora_alpha,                                    # 缩放系数
        lora_dropout=hparams.lora_dropout,                                # Dropout 概率
        layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,  # 限制变换层
        target_modules=hparams.target_modules,                            # 目标模块名（如 q_proj, v_proj）
    )

    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()  # 打印可训练参数量，便于调试

    adapter_name = "default"

    # ------------------------------------------------------------------ #
    # Step 3: 将标准 LoRA 升级为 CurvatureLora variant
    #   数据流: peft_model → 各 LoRA 模块的 lora_variant 字典被填充
    #           后续 forward 时，增量会被投影到安全子空间再叠加到输出
    # ------------------------------------------------------------------ #
    attach_curvature_lora_variant(peft_model, adapter_name=adapter_name)

    # ------------------------------------------------------------------ #
    # Step 4: 将 K-FAC 高曲率基向量注入各 CurvatureLora 模块
    #   数据流:
    #     layer_to_cov_cache_old[layer_name] → matched_cache
    #     matched_cache["Ua"] → Ua (d_in × d_in)   输入空间特征向量（升序）
    #     matched_cache["Ub"] → Ub (d_out × d_out) 输出空间特征向量（升序）
    #     ~mask_a → 高曲率（高能量）列索引
    #     Ua[:, ~mask_a] → U_in_bar  (d_in, k_in)  高曲率输入基
    #     Ub[:, ~mask_b] → U_out_bar (d_out, k_out) 高曲率输出基
    #   注入后 CurvatureLora.forward 可利用这两组基构建正交投影，
    #   将 ΔW 限制在 span(U_in_bar)⊥ ⊗ span(U_out_bar)⊥（安全子空间）
    # ------------------------------------------------------------------ #
    for name, module in peft_model.named_modules():
        # 仅处理同时具备 lora_A/lora_B 且已被升级为 CurvatureLora 的模块
        if not (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and hasattr(module, "in_features")
            and hasattr(module, "out_features")
        ):
            continue
        if adapter_name not in getattr(module, "lora_variant", {}):
            continue

        # 在缓存字典中查找与当前模块名最匹配的层
        matched_cache = None
        for layer_name, cache in layer_to_cov_cache_old.items():
            # 去掉尾部 ".weight" 后缀再做子串匹配
            clean = layer_name[: -len(".weight")] if layer_name.endswith(".weight") else layer_name
            if clean in name:
                matched_cache = cache
                break

        if matched_cache is None:
            continue  # 该模块无对应缓存，跳过（保持普通 LoRA 行为）

        # 提取基础权重的设备和数据类型，保证后续张量一致
        base_weight = module.base_layer.weight if hasattr(module, "base_layer") else module.weight
        dev = base_weight.device
        dtype = base_weight.dtype

        # Ua/Ub 的列按特征值升序排列；
        # mask==True  → 低能量（低曲率，安全方向）
        # mask==False → 高能量（高曲率，危险方向），即我们要避开的方向
        Ua = matched_cache["Ua"].to(device=dev, dtype=dtype)  # (d_in,  d_in)
        Ub = matched_cache["Ub"].to(device=dev, dtype=dtype)  # (d_out, d_out)
        mask_a = matched_cache["mask_a"].to(device=dev)       # BoolTensor，True=低曲率
        mask_b = matched_cache["mask_b"].to(device=dev)

        # 提取高曲率方向（~mask = False 列）
        U_in_bar  = Ua[:, ~mask_a]   # (d_in,  k_in)  —— 输入高曲率基
        U_out_bar = Ub[:, ~mask_b]   # (d_out, k_out) —— 输出高曲率基

        # 将基向量作为属性注入模块，CurvatureLora.forward 在运行时读取
        setattr(module, f"U_in_bar_{adapter_name}",  U_in_bar)
        setattr(module, f"U_out_bar_{adapter_name}", U_out_bar)

    # ------------------------------------------------------------------ #
    # Step 5: 构建普通 Adam 优化器
    #   只优化 lora_A / lora_B（基础权重已冻结）。
    #   CurvatureLora 的子空间约束通过 forward 结构实现，无需特殊优化器。
    # ------------------------------------------------------------------ #
    opt = torch.optim.Adam(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )

    # 记录训练前的旧数据 loss 作为基线
    if not getattr(hparams, "disable_old_loss_check", False):
        old_loss = calculate_old_loss(peft_model, tok, hparams)
        tracker.log(old_loss)

    # ------------------------------------------------------------------ #
    # Step 6: 训练循环
    #   数据流（单 batch）:
    #     [prompt_i + target_new_i]  （字符串拼接）
    #          │
    #          ▼ tokenize（padding=right, truncation, max_length）
    #     input_ids: (B, L)  attention_mask: (B, L)
    #          │
    #          ▼ 构造 labels（-100 遮蔽 pad 和 prompt 部分）
    #     labels:    (B, L)  仅 target 位置有效
    #          │
    #          ▼ peft_model forward（CurvatureLora 在内部做子空间投影）
    #     loss = CrossEntropy(logits[:, :-1], labels[:, 1:])
    #          │
    #          ▼ loss >= 1e-2 时反向传播 + Adam step
    #     lora_A, lora_B 参数更新（增量保持在安全子空间内）
    # ------------------------------------------------------------------ #
    loss_meter = AverageMeter()
    pbar = trange(hparams.num_steps)

    for it in pbar:
        loss_meter.reset()

        # 每轮随机打乱，减少顺序偏差
        random.shuffle(requests)
        texts   = [r["prompt"]     for r in requests]
        targets = [r["target_new"] for r in requests]

        for txt, tgt in zip(
            chunks(texts,   hparams.batch_size),
            chunks(targets, hparams.batch_size),
        ):
            # 拼接 prompt + target，形成完整的训练序列
            inputs_targets = [txt_ + tgt_ for txt_, tgt_ in zip(txt, tgt)]
            encodings = tok(
                inputs_targets,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=hparams.max_length,
            ).to(device)

            # 构造 labels：只在 target_new 对应 token 位置计算 loss
            labels = encodings["input_ids"].clone()
            labels[labels == tok.pad_token_id] = -100   # 忽略 padding
            for i, prompt in enumerate(txt):
                # 获取 prompt 部分的 token 长度，将其对应位置置 -100
                prompt_len = len(
                    tok(
                        prompt,
                        add_special_tokens=True,
                        truncation=True,
                        max_length=hparams.max_length,
                    )["input_ids"]
                )
                labels[i, :prompt_len] = -100  # 忽略 prompt token 的 loss

            opt.zero_grad(set_to_none=True)
            outputs = peft_model(**encodings, labels=labels)
            loss = outputs.loss

            loss_meter.update(loss.item(), n=labels.size(0))

            # loss 过小时跳过更新，避免无意义的微小梯度扰动
            if loss.item() >= 1e-2:
                loss.backward()
                opt.step()

        # 记录当前轮次的旧数据 loss 和编辑 loss
        if not getattr(hparams, "disable_old_loss_check", False):
            metrics = calculate_old_loss(peft_model, tok, hparams)
            metrics.update({"FT ParamLoRA Loss": loss_meter.avg})
            tracker.log(metrics)
        else:
            tracker.log({"FT ParamLoRA Loss": loss_meter.avg})

        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})
        # 收敛提前停止
        if loss_meter.avg < 1e-2:
            break

    # ------------------------------------------------------------------ #
    # Step 7: 合并 LoRA 权重并返回标准模型
    #   数据流: peft_model（带 CurvatureLora 适配器）
    #        → ΔW = (lora_alpha/r) * lora_B·lora_A 被吸收进 W
    #        → 去除 PEFT wrapper，返回与输入相同结构的 AutoModelForCausalLM
    # ------------------------------------------------------------------ #
    merged_model = peft_model.merge_and_unload()
    return merged_model
    
    