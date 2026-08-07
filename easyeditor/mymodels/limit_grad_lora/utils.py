import os
import math
from pathlib import Path
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from peft import LoraConfig, AdaLoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from copy import deepcopy

from .projected_lora_optimizer import ProjectedLoRAOptimizer
from ...models.rome.layer_stats import layer_stats_kfac_one_pass
from .mylora_hparams import MyLoRAHyperParams
from ...tools import ExperimentTracker

load_dotenv()
STATS_DIR = os.getenv("STATS_DIR")


def _is_llama_or_phi(model_name: str) -> bool:
    lower = model_name.lower()
    return "llama" in lower or "phi" in lower


def _resolve_cache_path(path_like: Optional[str]) -> Optional[Path]:
    if not path_like:
        return None
    path = Path(path_like)
    if path.is_absolute():
        return path
    if STATS_DIR:
        return Path(STATS_DIR) / path
    return path



def _compute_pretrain_kfac_stats(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_names: List[str],
    hparams: MyLoRAHyperParams,
    force_recompute: bool,
) -> Dict[str, Tuple]:
    return layer_stats_kfac_one_pass(
        model=model,
        tokenizer=tok,
        layer_names=layer_names,
        stats_dir=STATS_DIR,
        ds_name=hparams.mom2_dataset,
        to_collect=["mom2"],
        sample_size=hparams.mom2_n_samples,
        precision=hparams.mom2_dtype,
        force_recompute=force_recompute,
    )



def _load_kfac_stats_dict(cache_path: Path, layer_names: List[str],dtype:str,size:int) -> Dict[str, Tuple]:
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"KFAC cache directory not found: {cache_path}")
    if not cache_path.is_dir():
        raise NotADirectoryError(f"KFAC cache path is not a directory: {cache_path}")

    cache_dtype = getattr(torch, dtype)
    stats_dict = {}
    for layer_name in layer_names:
        
        filename=cache_path / f"{layer_name}_{dtype}_kfac{size}.npz"
        print(f"[_load_kfac_stats_dict]filepath:{filename}")
        if filename.exists():
            loaded = torch.load(filename, map_location='cpu')
            stats_dict[layer_name] = (loaded['A'].to(dtype=cache_dtype), loaded['B'].to(dtype=cache_dtype), loaded['N'])
        else:
            raise KeyError(f"KFAC cache {cache_path} missing layer {layer_name}")
    
    
    return stats_dict


def _save_kfac_stats_dict(
    cache_path: Path,
    stats_dict: Dict[str, Tuple],
    metadata: Optional[Dict] = None,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_kfac{size_suffix}.npz"
    filename = stats_dir / file_extension
    filename.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'A': A, 'B': B, 'N': total_tokens}, filename)


def _merge_kfac_stats_dicts(




    base_stats: Dict[str, Tuple],
    task_stats: Dict[str, Tuple],
    layer_names: List[str],
    task_weight: float,
    dtype:str
) -> Dict[str, Tuple]:
    task_weight = min(max(float(task_weight), 0.0), 1.0)
    base_weight = 1.0 - task_weight
    dtype = getattr(torch, dtype)
    merged = {}
    for layer_name in layer_names:
        A_base, B_base, n_base = base_stats[layer_name]
        A_task, B_task, n_task = task_stats[layer_name]
        A_mix = base_weight * A_base.to("cuda",dtype=dtype) + task_weight * A_task.to("cuda",dtype=dtype)
        B_mix = base_weight * B_base.to("cuda",dtype=dtype) + task_weight * B_task.to("cuda",dtype=dtype)
        merged[layer_name] = (
            A_mix.to("cpu"),
            B_mix.to("cpu"),
            n_base+n_task
        )
    return merged


def get_topk_indices_by_energy_ratio(eigenvalues: torch.Tensor, percent: float = 0.9):
    """
        根据累积能量比例选择高曲率方向。
        返回：
        k: 需要保留的高曲率方向数量
        idx: 高曲率方向对应的特征值下标
        threshold: 第 k 个特征值阈值
    """
    # 协方差矩阵特征值一定大于等于0
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    sorted_eigvals, sorted_idx = torch.sort(eigenvalues, descending=True)
    total_energy = torch.sum(sorted_eigvals)
    if total_energy <= 0:
        return 0, sorted_idx[:0], torch.tensor(0.0, device=eigenvalues.device)
    cumulative_energy = torch.cumsum(sorted_eigvals, dim=0)
    energy_ratio = cumulative_energy / total_energy
    k = torch.searchsorted(energy_ratio, percent).item() + 1
    idx = sorted_idx[:k]
    threshold = sorted_eigvals[k - 1]
    return k, idx, threshold


def compute_marginal_masks(
    Sa: torch.Tensor,
    Ua: torch.Tensor,
    Sb: torch.Tensor,
    Ub: torch.Tensor,
    energy_threshold: float,
    return_eigvals: bool = False,
) -> Tuple[torch.Tensor, ...]:
    k_in, idx_in, threshold_in = get_topk_indices_by_energy_ratio(Sa, percent=energy_threshold)
    k_out, idx_out, threshold_out = get_topk_indices_by_energy_ratio(Sb, percent=energy_threshold)

    mask_a = Ua[:, idx_in].contiguous()
    mask_b = Ub[:, idx_out].contiguous()
    eig_a = torch.clamp(Sa[idx_in], min=0.0).contiguous()
    eig_b = torch.clamp(Sb[idx_out], min=0.0).contiguous()

    print(
        f"  mask_a: {k_in}/{Sa.shape[0]} safe dirs, "
        f"threshold_a={threshold_in:.6f}"
    )
    print(
        f"  mask_b: {k_out}/{Sb.shape[0]} safe dirs, "
        f"threshold_b={threshold_out:.6f}"
    )
    if return_eigvals:
        return mask_a, mask_b, eig_a, eig_b
    return mask_a, mask_b


def build_lora_projection_cache(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: MyLoRAHyperParams,
    force_recompute: bool = False,
) -> Dict[str, Dict]:
    print("[build_lora_projection_cache] 计算各层KFac协方差统计...")
    layer_names = [hparams.rewrite_module_tmp.format(l) for l in hparams.layers]

    task_kfac_cache_path = _resolve_cache_path(
        getattr(hparams, "task_kfac_cache_path", None)
    )
    base_kfac_cache_path = _resolve_cache_path(
        getattr(hparams, "base_kfac_cache_path", None)
    )
    merged_kfac_cache_path = _resolve_cache_path(
        getattr(hparams, "merged_kfac_cache_path", None)
    )
    task_kfac_weight = float(getattr(hparams, "task_kfac_weight", 0.0))
    cache_dtype = getattr(hparams, "mom2_n_dtype", "float32")
    cache_size = int(getattr(hparams, "mom2_n_sample", 10000))

    if ( merged_kfac_cache_path is not None and not force_recompute):
        print(f"[build_lora_projection_cache] 加载已融合 KFAC 缓存: {merged_kfac_cache_path}")
        stats_dict = _load_kfac_stats_dict(merged_kfac_cache_path, layer_names,cache_dtype,cache_size)
    else:
        if base_kfac_cache_path is not None:
            print(f"[build_lora_projection_cache] 加载基础 KFAC 记忆: {base_kfac_cache_path}")
            stats_dict = _load_kfac_stats_dict(base_kfac_cache_path, layer_names,cache_dtype,cache_size)
        else:
            #张文智：修改成为两步法.加载两个KFAC矩阵
            stats_dict = _compute_pretrain_kfac_stats(
                model, tok, layer_names, hparams, force_recompute
            )


        if task_kfac_cache_path is not None and task_kfac_weight > 0:
            print(f"[build_lora_projection_cache] 加载下游任务 KFAC 缓存: {task_kfac_cache_path}")
            task_stats_dict = _load_kfac_stats_dict(task_kfac_cache_path, layer_names,cache_dtype,cache_size)
            stats_dict = _merge_kfac_stats_dicts(
                stats_dict,
                task_stats_dict,
                layer_names,
                task_kfac_weight,
                cache_dtype,
            )
            _save_kfac_stats_dict(
                merged_kfac_cache_path,
                stats_dict,
                
            )
            print(f"[build_lora_projection_cache] 已保存融合 KFAC 缓存: {merged_kfac_cache_path}")

    layer_to_proj_cache = {}
    for layer_num, layer_name in zip(hparams.layers, layer_names):
        A, B, _ = stats_dict.pop(layer_name)
        if not _is_llama_or_phi(hparams.model_name):
            A, B = B, A
        print("正在进行特征分解......")
        A = A.to("cuda",dtype=torch.float32)
        B = B.to("cuda",dtype=torch.float32)

        Sa, Ua = torch.linalg.eigh(A)  
        Sb, Ub = torch.linalg.eigh(B) 

        print(f"[SchemeA] 层 {layer_name} 边缘化掩码计算:")
        mask_a, mask_b, eig_a, eig_b = compute_marginal_masks(
            Sa, Ua, Sb, Ub, hparams.energy_threshold, return_eigvals=True
        )

        layer_to_proj_cache[layer_name] = {
            "Ua": Ua.cpu(),
            "Ub": Ub.cpu(),
            "mask_a": mask_a.cpu(),
            "mask_b": mask_b.cpu(),
            "eig_a": eig_a.cpu(),
            "eig_b": eig_b.cpu(),
        }
        del A, B, Sa, Sb, Ua, Ub
        torch.cuda.empty_cache()

    return layer_to_proj_cache


def _register_leak_rate_for_layer(module, layer_name_clean: str, base_dtype: torch.dtype) -> nn.Parameter:
    attr = "leak_rate_default"
    if not hasattr(module, attr):
        param = nn.Parameter(torch.tensor([-4.0], dtype=base_dtype))
        module.register_parameter(attr, param)
        print(f"[GradLoRA] {layer_name_clean}: 注册 leak_rate_default (init=-4.0)")
    return getattr(module, attr)


def map_proj_cache_to_lora_params(
    peft_model,
    layer_to_proj_cache: Dict[str, Dict],
) -> Tuple[Dict[torch.nn.Parameter, Dict], List[nn.Parameter]]:
    param_to_proj_cache: Dict[torch.nn.Parameter, Dict] = {}

    # layer_clean → leak_rate_param 的映射，确保同一层 A/B 共享同一个 leak 参数
    layer_to_leak: Dict[str, nn.Parameter] = {}

    for name, param in peft_model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_A" not in name and "lora_B" not in name:
            continue

        matched_layer = None
        for layer_name in layer_to_proj_cache:
            # 去掉末尾的.weight
            clean = layer_name[:-len(".weight")] if layer_name.endswith(".weight") else layer_name
            if clean in name:
                matched_layer = layer_name
                break

        if matched_layer is None:
            continue

        clean = matched_layer[:-len(".weight")] if matched_layer.endswith(".weight") else matched_layer

        # 找到对应的 LoRA module，注册或复用 leak_rate_default
        if clean not in layer_to_leak:
            # 从 peft_model 中找到该层的 module（用于 register_parameter）
            matched_module = None
            for mod_name, mod in peft_model.named_modules():
                if clean in mod_name and hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
                    matched_module = mod
                    break
            if matched_module is not None:
                base_weight = (
                    matched_module.base_layer.weight
                    if hasattr(matched_module, "base_layer")
                    else matched_module.weight
                )
                leak_param = _register_leak_rate_for_layer(matched_module, clean, base_weight.dtype)
                layer_to_leak[clean] = leak_param
            else:
                # fallback：创建一个游离 Parameter（不挂在 module 上）
                layer_to_leak[clean] = nn.Parameter(torch.tensor([-4.0], dtype=param.dtype))

        leak_rate_param = layer_to_leak[clean]
        cache = layer_to_proj_cache[matched_layer]

        if "lora_A" in name:
            # lora_A: (r, d_in)，右投影，使用 Ua + mask_a
            param_to_proj_cache[param] = {
                "Ua":              cache["Ua"],
                "mask_a":          cache["mask_a"],
                "eig_a":           cache["eig_a"],
                "leak_rate_param": leak_rate_param,
                "param_type":      "lora_A",
            }
        elif "lora_B" in name:
            # lora_B: (d_out, r)，左投影，使用 Ub + mask_b
            param_to_proj_cache[param] = {
                "Ub":              cache["Ub"],
                "mask_b":          cache["mask_b"],
                "eig_b":           cache["eig_b"],
                "leak_rate_param": leak_rate_param,
                "param_type":      "lora_B",
            }

    leak_params = list(layer_to_leak.values())
    print(
        f"[GradLoRA] 建立参数→投影映射，共 {len(param_to_proj_cache)} 个 LoRA 参数，"
        f"{len(leak_params)} 个 leak_rate 参数"
    )
    return param_to_proj_cache, leak_params


def wrap_model_and_build_projected_optimizer(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: MyLoRAHyperParams,
    force_recompute: bool = False,
):
    print("1、计算 KFac 矩阵...")
    layer_to_proj_cache = build_lora_projection_cache(
        model, tok, hparams, force_recompute
    )

    print("2、挂载lora")
    model.config.use_cache = False
    model.enable_input_require_grads()

    if hparams.lora_type == "lora":
        ConfigClass = LoraConfig
    elif hparams.lora_type == "adalora":
        ConfigClass = AdaLoraConfig
    else:
        raise ValueError(f"Unsupported lora_type: {hparams.lora_type}")

    peft_config = ConfigClass(
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


    print("3、将之前计算的结果注册到lora的各层")
    param_to_proj_cache, leak_params = map_proj_cache_to_lora_params(
        peft_model, layer_to_proj_cache
    )

    lora_params = [p for p in peft_model.parameters() if p.requires_grad
                   and not any(p is lp for lp in leak_params)]

    print("4、创建优化器")
    optimizer_lora = ProjectedLoRAOptimizer(
        params=lora_params,
        projection_cache_map=param_to_proj_cache,
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
        projection_mode=hparams.projection_mode,
        use_leak=hparams.use_leak,
        leak_rate=hparams.leak_rate,
        newton_damping=getattr(hparams, "newton_damping", 1e-3),
        use_dynamic_projection=getattr(hparams, "use_dynamic_projection", True),
        dynamic_projection_beta=getattr(hparams, "dynamic_projection_beta", 0.95),
        dynamic_projection_strength=getattr(hparams, "dynamic_projection_strength", 0.5),
        dynamic_projection_min_scale=getattr(hparams, "dynamic_projection_min_scale", 0.2),
    )

    optimizer_leak = torch.optim.Adam(leak_params, lr=hparams.lr * 2)

    print(
        f"[GradLoRA] 初始化完成：LoRA rank={hparams.lora_rank}，"
        f"投影模式={hparams.projection_mode}，"
        f"能量阈值={hparams.energy_threshold}，"
        f"leak_params 数量={len(leak_params)}"
    )

    return peft_model, optimizer_lora, optimizer_leak, layer_to_proj_cache



def apply_limit_grad_lora_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MyLoRAHyperParams,
    return_orig_weights: bool = False,
    keep_original_weight: bool = False,
    **kwargs,
) -> AutoModelForCausalLM:
    if tok.padding_side != "right":
        tok.padding_side = "right"

    for i, request in enumerate(requests):
        if request["target_new"] and request["target_new"][0] != " ":
            requests[i]["target_new"] = " " + request["target_new"]

    peft_model, optimizer_lora, optimizer_leak, _ = wrap_model_and_build_projected_optimizer(
        model, tok, hparams
    )

    device = next(peft_model.parameters()).device

    texts = [
        r["prompt"].format(r.get("subject", "")) if "{}" in r["prompt"] else r["prompt"]
        for r in requests
    ]
    targets = [r["target_new"] for r in requests]

    peft_model.train()
    for step in range(hparams.num_steps):
        total_loss = 0.0
        for txt_batch, tgt_batch in zip(
            _chunks(texts, hparams.batch_size),
            _chunks(targets, hparams.batch_size),
        ):
            optimizer_lora.zero_grad()
            optimizer_leak.zero_grad()

            loss = _compute_loss(peft_model, tok, txt_batch, tgt_batch, device, hparams)

            if loss.item() >= 1e-3:
                loss.backward()
                optimizer_lora.step()
                optimizer_leak.step()

            total_loss += loss.item()

        num_batches = max(1, math.ceil(len(texts) / hparams.batch_size))
        avg_loss = total_loss / num_batches
        ExperimentTracker.log({"LOSS": avg_loss})
        print(f"[GradLoRA] Step {step+1}/{hparams.num_steps}  loss={avg_loss:.4f}")
        if avg_loss < 1e-3:
            print("[GradLoRA] 损失收敛，提前结束训练")
            break

    peft_model = peft_model.merge_and_unload()
    return peft_model


def _compute_loss(
    model,
    tok: AutoTokenizer,
    texts: List[str],
    targets: List[str],
    device: torch.device,
    hparams: MyLoRAHyperParams,
) -> torch.Tensor:
    """
    计算编辑损失，参照 crispedit.py 的 execute_ft 训练循环：
      - 拼接 prompt + target_new
      - prompt 部分 label 设为 -100（只对 target 计算 loss）
      - padding token 的 label 设为 -100
    """
    inputs_targets = [t + tg for t, tg in zip(texts, targets)]
    encodings = tok(
        inputs_targets,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
    ).to(device)

    labels = encodings["input_ids"].clone()
    # padding 位置不计 loss
    labels[labels == tok.pad_token_id] = -100
    # prompt 部分不计 loss
    for i, prompt in enumerate(texts):
        prompt_len = len(
            tok(prompt, add_special_tokens=True, truncation=True,
                max_length=hparams.max_length)["input_ids"]
        )
        labels[i, :prompt_len] = -100

    return model(**encodings, labels=labels).loss


def _chunks(lst: List, n: int):
    """将列表按大小 n 切分，参照 utils.py 的 chunks 函数"""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]