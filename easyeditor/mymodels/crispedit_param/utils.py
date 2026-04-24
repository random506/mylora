import gc
import random
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os

from .CrispEditParam_hparams import CrispEditParamHyperParams
from ...models.rome.layer_stats import (
    layer_stats_kfac_one_pass,
)

load_dotenv()
STATS_DIR = os.getenv("STATS_DIR")


def get_rank_and_threshold_by_energy_ratio(eigenvalues: torch.Tensor, percent: float = 0.9):
    """按累积能量比例确定保留的特征值数量及对应阈值。"""
    total_energy = torch.sum(eigenvalues)
    sorted_eigvals, _ = torch.sort(eigenvalues, descending=True)
    cumulative_energy = torch.cumsum(sorted_eigvals, dim=0)
    energy_ratio = cumulative_energy / total_energy
    rank = torch.searchsorted(energy_ratio, percent).item() + 1
    threshold = sorted_eigvals[rank - 1] if rank - 1 < len(sorted_eigvals) else 0.0
    return rank, threshold


def calculate_projection_cache_with_kfac(A: torch.Tensor, B: torch.Tensor, energy_threshold: float = 0.9) -> Dict:
    # 特征分解、计算掩码矩阵
    Sa, Ua = torch.linalg.eigh(A)  
    Sb, Ub = torch.linalg.eigh(B)

    M = torch.outer(Sa, Sb)          # (d_in, d_out)，联合曲率
    rank, null_threshold = get_rank_and_threshold_by_energy_ratio(M.view(-1), percent=energy_threshold)
    M = (M < null_threshold).float()  # 1.0 = 安全方向，0.0 = 危险方向

    print(
        f"[CrispEditParam] safe rank={rank}/{Sa.shape[0] * Sb.shape[0]}, "
        f"null_threshold={null_threshold:.6f}"
    )
    return {"Ua": Ua, "Ub": Ub, "M": M}



def calculate_cov_cache_with_old_data(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: CrispEditParamHyperParams,
    force_recompute: bool = False,
) -> Dict[str, Dict]:
    # 计算协方差矩阵
    if hparams.no_crisp:
        return None

    layer_name_map = {
        layer_num: hparams.rewrite_module_tmp.format(layer_num)
        for layer_num in hparams.layers
    }
    target_layers = list(layer_name_map.values())

    stats_dict = layer_stats_kfac_one_pass(
        model=model, tokenizer=tok,
        layer_names=target_layers, stats_dir=STATS_DIR,
        ds_name=hparams.mom2_dataset, to_collect=["mom2"],
        sample_size=hparams.mom2_n_samples,
        precision=hparams.mom2_dtype,
        force_recompute=force_recompute,
    )

    layer_to_cov_cache = {}
    for layer_num in hparams.layers:
        layer_name = layer_name_map[layer_num]
        A, B, num_samples = stats_dict.pop(layer_name)
        layer_to_cov_cache[layer_name] = {
            "A": A.to("cpu", dtype=torch.float32),
            "B": B.to("cpu", dtype=torch.float32),
            "num_samples": num_samples,
        }
        del A, B

    return layer_to_cov_cache



def combine_layer_to_cov_caches(layer_to_cov_caches: List[Dict[str, Dict]]) -> Dict[str, Dict]:
    # 针对于num_samples，可对协方差矩阵求均值
    if len(layer_to_cov_caches) == 1:
        return layer_to_cov_caches[0]

    combined = {}
    for layer_name in layer_to_cov_caches[0].keys():
        A_list  = [c[layer_name]["A"] for c in layer_to_cov_caches]
        B_list  = [c[layer_name]["B"] for c in layer_to_cov_caches]
        ns_list = [c[layer_name]["num_samples"] for c in layer_to_cov_caches]
        total   = sum(ns_list)
        combined[layer_name] = {
            "A": sum(A * n for A, n in zip(A_list, ns_list)) / total,
            "B": sum(B * n for B, n in zip(B_list, ns_list)) / total,
            "num_samples": total,
        }
        print(f"Combined samples {ns_list} for {layer_name}")
    return combined



def calculate_projection_caches_from_cov_caches(
    model: AutoModelForCausalLM,
    hparams: CrispEditParamHyperParams,
    layer_to_cov_caches: Dict[str, Dict],
    energy_threshold: Optional[float] = None,
) -> Dict[torch.nn.Parameter, Dict]:
    """
    对每个目标层计算投影缓存，返回 {weight_param → {Ua, Ub, M}}。
    键为参数对象本身，与 ProjectedAdam 的接口一致。
    """
    energy_threshold = energy_threshold or hparams.energy_threshold
    weights = get_weights(model, hparams, bias=False)
    weight_to_projection_cache = {}

    for layer_name, cov_cache in layer_to_cov_caches.items():
        A = cov_cache["A"].to(model.device)
        B = cov_cache["B"].to(model.device)

        # 非 Llama/phi 类模型需要交换 A/B（与原 crispedit 保持一致）
        if hparams.model_name.lower() not in ["llama3-8b", "phi-1.5"]:
            A, B = B, A

        proj_cache = calculate_projection_cache_with_kfac(A, B, energy_threshold)

        # 统一类型（M 保留 float，Ua/Ub 转换为模型 dtype）
        model_dtype = next(model.parameters()).dtype
        proj_cache["Ua"] = proj_cache["Ua"].to(model.device, dtype=model_dtype)
        proj_cache["Ub"] = proj_cache["Ub"].to(model.device, dtype=model_dtype)
        proj_cache["M"]  = proj_cache["M"] .to(model.device, dtype=model_dtype)

        # 键为权重参数本身
        weight_param = weights[layer_name]
        weight_to_projection_cache[weight_param] = proj_cache

    return weight_to_projection_cache




def get_weights(
    model: AutoModelForCausalLM,
    hparams: CrispEditParamHyperParams,
    bias: bool,
    to_cpu: bool = False,
) -> Dict[str, torch.Tensor]:
    """提取目标层的权重参数字典（不含 bias），可选择拷贝到 CPU。"""
    # 将会返回目标层及其对应参数的字典
    return {
        n: (p.detach().cpu().clone() if to_cpu else p)
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n and "bias" not in n
    }


def cache_weights_to_cpu(weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """将权重字典中所有张量拷贝到 CPU 并 detach，用于后续变化检测。"""
    return {name: param.detach().cpu().clone() for name, param in weights.items()}


def is_weights_changed(
    current_weights: Dict[str, torch.Tensor],
    cached_weights: Dict[str, torch.Tensor],
    threshold: float,
) -> bool:
    """检测模型权重相对于缓存快照是否发生显著变化（相对范数超过阈值）。"""
    for name, param in current_weights.items():
        cached_param = cached_weights[name]
        change = torch.norm(param.detach().cpu() - cached_param) / (torch.norm(cached_param) + 1e-8)
        if change > threshold:
            print(f"Weight {name} changed by {change:.4f}, exceeding threshold {threshold}.")
            return True
    return False


def recalculate_cov_cache_if_weights_changed(
    model, tok,
    hparams: CrispEditParamHyperParams,
    current_weights_cpu: Dict,
    layer_to_cov_cache: Dict,
) -> Tuple[Dict, Dict, bool]:
    """若权重变化超过阈值则重新计算协方差缓存，返回 (新权重快照, 新协方差缓存, 是否重算)。"""
    if not hparams.recalculate_cache or hparams.no_crisp:
        return current_weights_cpu, layer_to_cov_cache, False

    weights = get_weights(model, hparams, bias=True)
    if not is_weights_changed(weights, current_weights_cpu, hparams.recalculate_weight_threshold):
        return current_weights_cpu, layer_to_cov_cache, False

    del layer_to_cov_cache, weights
    gc.collect()
    torch.cuda.empty_cache()

    layer_to_cov_cache = calculate_cov_cache_with_old_data(model, tok, hparams, force_recompute=True)
    weights = get_weights(model, hparams, bias=True)
    current_weights_cpu = cache_weights_to_cpu(weights)

    return current_weights_cpu, layer_to_cov_cache, True


def update_model_and_tokenizer_with_appropriate_padding_token(model, tokenizer, hparams):
    """为模型和分词器配置合适的 padding token（Qwen 复用 eos，其余模型添加 [PAD]）。"""
    if "Qwen" in hparams.model_name:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer



def _chunks(lst: List, n: int):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def execute_crispedit_param(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: "CrispEditParamHyperParams",
    **kwargs: Any,  # 支持 tracker=... 传入外部日志对象
) -> AutoModelForCausalLM:
    """
    CrispEditParam 主编辑接口：直接对目标层原始权重做曲率投影约束的梯度下降。
    """
    print("1、进入执行函数")
    tracker = kwargs.get("tracker", None)
    device  = model.device

    if tok.padding_side != "right":
        tok.padding_side = "right"

    # 增加结束字符，有些方法是在外部添加!
    # model, tok = update_model_and_tokenizer_with_appropriate_padding_token(model, tok, hparams)

    requests = deepcopy(requests)
    for i, req in enumerate(requests):
        if req["target_new"] and req["target_new"][0] != " ":
            requests[i]["target_new"] = " " + req["target_new"]

    # 1、根据配置文件，冻结其余层参数
    print("2、冻结非训练层参数")
    weights = get_weights(model, hparams, bias=False)
    for name, param in model.named_parameters():
        param.requires_grad = name in weights

    # 2、根据rome算法中的函数，计算协方差矩阵
    print("3、计算协方差矩阵")
    layer_to_cov_cache_old = calculate_cov_cache_with_old_data(model, tok, hparams)

    # 3、对协方差进行特征分解，得到投影基 {param → {Ua, Ub, M}}
    print("4、求解掩码矩阵")
    combined_cov_cache = combine_layer_to_cov_caches([layer_to_cov_cache_old])
    weight_to_projection_cache = calculate_projection_caches_from_cov_caches(
        model, hparams, combined_cov_cache
    )

    # 4、创建优化器
    weights_list = list(weights.values())
    opt = torch.optim.Adam(
        weights_list,
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,#正则化系数
    )
    current_weights_cpu = cache_weights_to_cpu(weights)

    # ── 阶段 4：训练循环 ──────────────────────────────────────────────────
    texts   = [r["prompt"]     for r in requests]
    targets = [r["target_new"] for r in requests]

    for step in range(hparams.num_steps):
        loss_sum, loss_cnt = 0.0, 0

        # 每轮打乱，降低顺序偏差
        paired = list(zip(texts, targets))
        random.shuffle(paired)
        texts_shuffled, targets_shuffled = zip(*paired)

        for txt, tgt in zip(
            _chunks(list(texts_shuffled),   hparams.batch_size),
            _chunks(list(targets_shuffled), hparams.batch_size),
        ):
            inputs_targets = [t + g for t, g in zip(txt, tgt)]
            encodings = tok(
                inputs_targets,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=hparams.max_length,
            ).to(device)

            # 只对 target 部分计算 loss，prompt 位置置 -100
            labels = encodings["input_ids"].clone()
            labels[labels == tok.pad_token_id] = -100
            for i, prompt in enumerate(txt):
                prompt_len = len(
                    tok(prompt, add_special_tokens=True,
                        truncation=True, max_length=hparams.max_length)["input_ids"]
                )
                labels[i, :prompt_len] = -100

            opt.zero_grad(set_to_none=True)
            loss = model(**encodings, labels=labels).loss
            loss_sum += loss.item() * labels.size(0)
            loss_cnt += labels.size(0)

            if loss.item() >= 1e-2:
                loss.backward()

                # 记录 step 前的权重快照，用于计算实际 ΔW
                w_before = {p: p.data.clone() for p in weights_list}

                opt.step()

                with torch.no_grad():
                    for param, proj in weight_to_projection_cache.items():
                        if param.data.ndim != 2:
                            continue
                        U_in  = proj["Ua"].to(device=param.device, dtype=param.dtype)
                        U_out = proj["Ub"].to(device=param.device, dtype=param.dtype)
                        M     = proj["M"] .to(device=param.device, dtype=param.dtype)

                        delta_W = param.data - w_before[param]

                        C = U_out.T @ delta_W @ U_in
                        C = C * M.T
                        param.data = w_before[param] + U_out @ C @ U_in.T

                '''
                # 更新之后模型代表的分布发生改变，需要更新K-FAC矩阵进行后续投影

                current_weights_cpu, layer_to_cov_cache_old, recalculated = \
                    recalculate_cov_cache_if_weights_changed(
                        model, tok, hparams, current_weights_cpu, layer_to_cov_cache_old
                    )
                if recalculated:
                    new_combined = combine_layer_to_cov_caches([layer_to_cov_cache_old])
                    weight_to_projection_cache = calculate_projection_caches_from_cov_caches(
                        model, hparams, new_combined
                    )
                '''

        avg_loss = loss_sum / max(loss_cnt, 1)
        print(f"[CrispEditParam] step {step+1}/{hparams.num_steps}  loss={avg_loss:.4f}")
        if tracker is not None:
            tracker.log({"CrispEditParam/loss": avg_loss, "CrispEditParam/step": step + 1})
        if avg_loss < 1e-2:
            print("[CrispEditParam] 提前收敛，停止训练")
            break

    return model
