import gc
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from dotenv import load_dotenv
from peft import AdaLoraConfig, LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from easyeditor.models.crispedit.projected_adam import ProjectedAdam
from easyeditor.models.crispedit.projected_sgd import ProjectedSGD
from ..rome.layer_stats import (
    calculate_cache_loss,
    calculate_request_loss,
    layer_stats_kfac,
    layer_stats_kfac_one_pass,
    layer_stats_kfac_with_txt_tgt,
)
from .CrispEdit_hparams import CrispEditHyperParams

load_dotenv()
STATS_DIR = os.getenv("STATS_DIR")


def _ensure_crispedit_defaults(hparams) -> None:
    defaults = {
        "perform_lora": False,
        "no_crisp": False,
        "recalculate_cache": False,
        "recalculate_weight_threshold": 0.01,
        "edit_n_samples": 10,
        "edit_cache_style": "new",
        "disable_old_loss_check": False,
    }
    for name, value in defaults.items():
        if not hasattr(hparams, name):
            setattr(hparams, name, value)


def _is_llama_or_phi(model_name: str) -> bool:
    lower = str(model_name).lower()
    return "llama" in lower or "phi" in lower or "qwen" in lower


def _model_device(model) -> torch.device:
    return getattr(model, "device", next(model.parameters()).device)


def _resolve_cache_path(path_like: Optional[str]) -> Optional[Path]:
    if path_like in (None, "", "null", "None"):
        return None
    path = Path(path_like)
    if path.is_absolute():
        return path
    if STATS_DIR:
        return Path(STATS_DIR) / path
    return path


def _layer_names(hparams) -> List[str]:
    return [hparams.rewrite_module_tmp.format(layer) for layer in hparams.layers]


def _cache_dtype_name(hparams) -> str:
    return getattr(hparams, "mom2_n_dtype", getattr(hparams, "mom2_dtype", "float32"))


def _cache_sample_size(hparams) -> int:
    return int(getattr(hparams, "mom2_n_sample", getattr(hparams, "mom2_n_samples", 10000)))


def _normalize_kfac_entry(entry, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if isinstance(entry, (tuple, list)):
        if len(entry) < 2:
            raise ValueError("KFAC tuple entries must contain at least A and B.")
        A, B = entry[0], entry[1]
        num_samples = entry[2] if len(entry) > 2 else 0
    elif isinstance(entry, dict):
        A = entry.get("A")
        B = entry.get("B")
        num_samples = entry.get("N", entry.get("num_samples", entry.get("n", 0)))
    else:
        raise TypeError(f"Unsupported KFAC cache entry type: {type(entry)}")

    if A is None or B is None:
        raise ValueError("KFAC cache entry must contain A and B tensors.")

    return A.to("cpu", dtype=dtype), B.to("cpu", dtype=dtype), int(num_samples)


def _load_torch_or_npz(path: Path):
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        import numpy as np

        loaded = np.load(path, allow_pickle=True)
        return {key: torch.from_numpy(loaded[key]) for key in loaded.files}


def _load_kfac_stats_dict(
    cache_path: Path,
    layer_names: List[str],
    dtype_name: str,
    sample_size: int,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, int]]:
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"KFAC cache path not found: {cache_path}")

    dtype = getattr(torch, dtype_name)
    stats_dict = {}

    if cache_path.is_dir():
        for layer_name in layer_names:
            candidates = [
                cache_path / f"{layer_name}_{dtype_name}_kfac_{sample_size}.npz",
                cache_path / f"{layer_name}_{dtype_name}_kfac{sample_size}.npz",
                cache_path / f"{layer_name}_{dtype_name}_kfac.npz",
                cache_path / f"{layer_name}.pt",
            ]
            filename = next((p for p in candidates if p.exists()), None)
            if filename is None:
                raise KeyError(f"KFAC cache {cache_path} missing layer {layer_name}")
            loaded = _load_torch_or_npz(filename)
            stats_dict[layer_name] = _normalize_kfac_entry(loaded, dtype)
        return stats_dict

    loaded = _load_torch_or_npz(cache_path)
    if isinstance(loaded, dict) and "stats" in loaded:
        loaded = loaded["stats"]

    if isinstance(loaded, dict) and "A" in loaded and "B" in loaded:
        if len(layer_names) != 1:
            raise ValueError(
                f"Single-layer KFAC file {cache_path} cannot be mapped to "
                f"{len(layer_names)} layers."
            )
        return {layer_names[0]: _normalize_kfac_entry(loaded, dtype)}

    for layer_name in layer_names:
        if layer_name not in loaded:
            raise KeyError(f"KFAC file {cache_path} missing layer {layer_name}")
        stats_dict[layer_name] = _normalize_kfac_entry(loaded[layer_name], dtype)
    return stats_dict


def _compute_pretrain_kfac_stats(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_names: List[str],
    hparams,
    force_recompute: bool,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, int]]:
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


def _build_cov_cache_from_hparams(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams,
    force_recompute: bool = False,
) -> Dict[str, Dict]:
    layer_names = _layer_names(hparams)
    dtype_name = _cache_dtype_name(hparams)
    sample_size = _cache_sample_size(hparams)

    base_kfac_cache_path = _resolve_cache_path(getattr(hparams, "base_kfac_cache_path", None))
    task_kfac_cache_path = _resolve_cache_path(getattr(hparams, "task_kfac_cache_path", None))
    task_stats_dict = None

    if base_kfac_cache_path is not None:
        print(f"[CrispEdit] Loading base KFAC cache: {base_kfac_cache_path}")
        stats_dict = _load_kfac_stats_dict(
            base_kfac_cache_path, layer_names, dtype_name, sample_size
        )
    else:
        print("[CrispEdit] Computing/loading pretrain KFAC stats.")
        stats_dict = _compute_pretrain_kfac_stats(
            model, tok, layer_names, hparams, force_recompute
        )

    if task_kfac_cache_path is not None:
        print(f"[CrispEdit] Loading task KFAC cache: {task_kfac_cache_path}")
        task_stats_dict = _load_kfac_stats_dict(
            task_kfac_cache_path, layer_names, dtype_name, sample_size
        )

    layer_to_cov_cache = {}
    for layer_name, (A, B, n) in stats_dict.items():
        cov_cache = {
            "A": A.to("cpu", dtype=torch.float32),
            "B": B.to("cpu", dtype=torch.float32),
            "num_samples": n,
        }
        if task_stats_dict is not None and layer_name in task_stats_dict:
            task_A, task_B, task_n = task_stats_dict[layer_name]
            cov_cache.update(
                {
                    "task_A": task_A.to("cpu", dtype=torch.float32),
                    "task_B": task_B.to("cpu", dtype=torch.float32),
                    "task_num_samples": task_n,
                }
            )
        layer_to_cov_cache[layer_name] = cov_cache
    return layer_to_cov_cache


def _first_hparam(hparams, names: List[str]):
    for name in names:
        value = getattr(hparams, name, None)
        if value not in (None, "", "null", "None"):
            return value
    return None


def _use_second_projection(hparams) -> bool:
    for name in (
        "use_second_projection",
        "use_two_projection",
        "use_double_projection",
        "two_projection",
        "use_additional_projection",
    ):
        if hasattr(hparams, name):
            return bool(getattr(hparams, name))
    return True


def _load_additional_cov_cache_from_hparams(hparams) -> Optional[Dict[str, Dict]]:
    path_like = _first_hparam(
        hparams,
        [
            "additional_kfac_cache_path",
            "second_kfac_cache_path",
            "second_projection_kfac_cache_path",
        ],
    )
    cache_path = _resolve_cache_path(path_like)
    if cache_path is None:
        return None

    layer_names = _layer_names(hparams)
    stats_dict = _load_kfac_stats_dict(
        cache_path, layer_names, _cache_dtype_name(hparams), _cache_sample_size(hparams)
    )
    print(f"[CrispEdit] Loading second-projection KFAC cache: {cache_path}")
    return {
        layer_name: {
            "A": A.to("cpu", dtype=torch.float32),
            "B": B.to("cpu", dtype=torch.float32),
            "num_samples": n,
        }
        for layer_name, (A, B, n) in stats_dict.items()
    }


def _normalize_projection_entry(entry) -> Dict:
    if not isinstance(entry, dict):
        raise TypeError(f"Unsupported projection cache entry type: {type(entry)}")

    cache = {}
    for key in (
        "mask_a",
        "mask_b",
        "eig_a",
        "eig_b",
        "task_mask_a",
        "task_mask_b",
        "task_eig_a",
        "task_eig_b",
    ):
        if key in entry:
            cache[key] = entry[key].to("cpu", dtype=torch.float32)

    has_base = all(key in cache for key in ("mask_a", "mask_b", "eig_a", "eig_b"))
    has_task = all(
        key in cache
        for key in ("task_mask_a", "task_mask_b", "task_eig_a", "task_eig_b")
    )
    if not has_base and not has_task:
        raise ValueError(
            "Projection cache entry must contain mask_a/mask_b/eig_a/eig_b "
            "or task_mask_a/task_mask_b/task_eig_a/task_eig_b."
        )
    return cache


def _load_layer_projection_cache(
    cache_path: Path,
    hparams,
) -> Dict[str, Dict]:
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Projection cache path not found: {cache_path}")

    layer_names = _layer_names(hparams)
    if cache_path.is_dir():
        layer_to_projection_cache = {}
        for layer_name in layer_names:
            clean = layer_name[:-len(".weight")] if layer_name.endswith(".weight") else layer_name
            candidates = [
                cache_path / f"{layer_name}_projection.pt",
                cache_path / f"{clean}_projection.pt",
                cache_path / f"{layer_name}.pt",
                cache_path / f"{clean}.pt",
            ]
            filename = next((p for p in candidates if p.exists()), None)
            if filename is None:
                raise KeyError(f"Projection cache {cache_path} missing layer {layer_name}")
            layer_to_projection_cache[layer_name] = _normalize_projection_entry(
                torch.load(filename, map_location="cpu")
            )
        return layer_to_projection_cache

    loaded = torch.load(cache_path, map_location="cpu")
    if isinstance(loaded, dict) and "projection_cache" in loaded:
        loaded = loaded["projection_cache"]

    if isinstance(loaded, dict) and (
        ("mask_a" in loaded and "mask_b" in loaded)
        or ("task_mask_a" in loaded and "task_mask_b" in loaded)
    ):
        if len(layer_names) != 1:
            raise ValueError(
                f"Single-layer projection cache {cache_path} cannot be mapped to "
                f"{len(layer_names)} layers."
            )
        return {layer_names[0]: _normalize_projection_entry(loaded)}

    return {
        layer_name: _normalize_projection_entry(loaded[layer_name])
        for layer_name in layer_names
    }


def _map_layer_projection_cache_to_weights(
    model,
    hparams,
    layer_to_projection_cache: Dict[str, Dict],
) -> Dict[torch.nn.Parameter, Dict]:
    weights = get_weights(model, hparams, bias=False)
    return {
        _find_weight_for_layer(weights, layer_name): cache
        for layer_name, cache in layer_to_projection_cache.items()
    }


def _prefix_task_projection_cache(cache_map: Dict[torch.nn.Parameter, Dict]) -> Dict:
    prefixed = {}
    for param, cache in cache_map.items():
        new_cache = {}
        for key, value in cache.items():
            if key.startswith("task_"):
                new_cache[key] = value
            elif key in ("mask_a", "mask_b", "eig_a", "eig_b"):
                new_cache[f"task_{key}"] = value
        prefixed[param] = new_cache
    return prefixed


def _merge_projection_cache_maps(primary: Optional[Dict], task: Optional[Dict]) -> Optional[Dict]:
    if primary is None:
        primary = {}
    if task is None:
        return primary
    for param, task_cache in task.items():
        primary.setdefault(param, {}).update(task_cache)
    return primary


def _load_projection_cache_map_from_hparams(
    model,
    hparams,
    names: List[str],
) -> Optional[Dict[torch.nn.Parameter, Dict]]:
    path_like = _first_hparam(hparams, names)
    cache_path = _resolve_cache_path(path_like)
    if cache_path is None:
        return None
    print(f"[CrispEdit] Loading projection cache: {cache_path}")
    layer_to_projection_cache = _load_layer_projection_cache(cache_path, hparams)
    return _map_layer_projection_cache_to_weights(model, hparams, layer_to_projection_cache)


def get_topk_indices_by_energy_ratio(eigenvalues: torch.Tensor, percent: float = 0.9):
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


def get_rank_and_threshold_by_energy_ratio(eigenvalues, percent=0.9):
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    sorted_eigvals, _ = torch.sort(eigenvalues, descending=True)
    total_energy = torch.sum(sorted_eigvals)
    if total_energy <= 0:
        return 0, torch.tensor(0.0, device=eigenvalues.device)

    cumulative_energy = torch.cumsum(sorted_eigvals, dim=0)
    energy_ratio = cumulative_energy / total_energy
    rank = torch.searchsorted(energy_ratio, percent).item() + 1
    threshold = sorted_eigvals[rank - 1] if rank - 1 < len(sorted_eigvals) else sorted_eigvals[-1]
    return rank, threshold


def calculate_projection_cache_with_kfac(A, B, energy_threshold=0.9):
    A = A.to("cuda",dtype=torch.float32)
    B = B.to("cuda",dtype=torch.float32)
    Sa, Ua = torch.linalg.eigh(A)
    Sb, Ub = torch.linalg.eigh(B)

    k_in, idx_in, threshold_in = get_topk_indices_by_energy_ratio(
        Sa, percent=energy_threshold
    )
    k_out, idx_out, threshold_out = get_topk_indices_by_energy_ratio(
        Sb, percent=energy_threshold
    )

    mask_a = Ua[:, idx_in].contiguous()
    mask_b = Ub[:, idx_out].contiguous()
    eig_a = torch.clamp(Sa[idx_in], min=0.0).contiguous()
    eig_b = torch.clamp(Sb[idx_out], min=0.0).contiguous()

    print(
        f"[CrispEdit] mask_a={k_in}/{Sa.shape[0]}, "
        f"threshold_a={float(threshold_in):.6f}; "
        f"mask_b={k_out}/{Sb.shape[0]}, "
        f"threshold_b={float(threshold_out):.6f}"
    )
    return {
        "mask_a": mask_a.cpu(),
        "mask_b": mask_b.cpu(),
        "eig_a": eig_a.cpu(),
        "eig_b": eig_b.cpu(),
    }


def get_cov_ab(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    force_recompute: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    A, B = layer_stats_kfac(
        model,
        tok,
        layer_name,
        STATS_DIR,
        mom2_dataset,
        to_collect=["mom2"],
        sample_size=mom2_n_samples,
        precision=mom2_dtype,
        force_recompute=force_recompute,
    )
    return A, B


def calculate_projection_cache_by_layer(model, tok, layer, hparams, force_recompute):
    layer_name = hparams.rewrite_module_tmp.format(layer)
    A, B = get_cov_ab(
        model,
        tok,
        layer_name,
        hparams.mom2_dataset,
        hparams.mom2_n_samples if not force_recompute else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
    )

    if not _is_llama_or_phi(hparams.model_name):
        A, B = B, A

    P_cache = calculate_projection_cache_with_kfac(
        A, B, energy_threshold=hparams.energy_threshold
    )
    model_dtype = next(model.parameters()).dtype
    return {
        key: value.to(device=_model_device(model), dtype=model_dtype)
        for key, value in P_cache.items()
    }


def get_weights(
    model: AutoModelForCausalLM,
    hparams: CrispEditHyperParams,
    bias: bool,
    to_cpu: bool = False,
) -> Dict[str, torch.Tensor]:
    bias = False
    return {
        n: (p.detach().cpu().clone() if to_cpu else p)
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n and (bias or ("bias" not in n))
    }


def calculate_cov_cache_with_old_data(model, tok, hparams, force_recompute=False) -> Dict[str, Dict]:
    _ensure_crispedit_defaults(hparams)
    if getattr(hparams, "no_crisp", False):
        return None
    return _build_cov_cache_from_hparams(model, tok, hparams, force_recompute)


def calculate_cov_cache_with_request(txt, tgt, model, tok, hparams):
    _ensure_crispedit_defaults(hparams)
    if getattr(hparams, "no_crisp", False):
        return None

    cov_stats_dict = layer_stats_kfac_with_txt_tgt(
        model,
        tok,
        layer_names=_layer_names(hparams),
        txt=txt,
        tgt=tgt,
        precision=hparams.mom2_dtype,
        sample_size=getattr(hparams, "edit_n_samples", 10),
        to_collect=["mom2"],
        add_pretrain_data=(getattr(hparams, "edit_cache_style", "new") == "mix"),
        pretrain_sample_size=hparams.mom2_n_samples,
    )

    layer_to_cov_cache = {}
    for layer_name in _layer_names(hparams):
        A, B, num_samples = cov_stats_dict.pop(layer_name)
        layer_to_cov_cache[layer_name] = {
            "A": A.to("cpu", dtype=torch.float32),
            "B": B.to("cpu", dtype=torch.float32),
            "num_samples": num_samples,
        }
        del A, B
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return layer_to_cov_cache


def cache_weights_to_cpu(weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not isinstance(weights, dict):
        raise ValueError("Input must be a dict of tensors.")
    return {name: param.detach().cpu().clone() for name, param in weights.items()}


def is_weights_changed(current_weights, cached_weights, threshold: float) -> bool:
    for name, param in current_weights.items():
        cached_param = cached_weights[name]
        denom = torch.norm(cached_param).clamp(min=1e-8)
        change = torch.norm(param.detach().cpu() - cached_param) / denom
        if change > threshold:
            print(f"Weight {name} changed by {change:.4f}, exceeding threshold {threshold}.")
            return True
    return False


def recalculate_cov_cache_if_weights_changed(
    model,
    tok,
    hparams,
    current_weights_cpu,
    layer_to_cov_cache,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict], bool]:
    _ensure_crispedit_defaults(hparams)
    if (
        not getattr(hparams, "recalculate_cache", False)
        or getattr(hparams, "no_crisp", False)
        or current_weights_cpu is None
    ):
        return current_weights_cpu, layer_to_cov_cache, False

    weights = get_weights(model, hparams, bias=True)
    threshold = getattr(hparams, "recalculate_weight_threshold", 0.01)
    if not is_weights_changed(weights, current_weights_cpu, threshold):
        return current_weights_cpu, layer_to_cov_cache, False

    del layer_to_cov_cache, weights
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    layer_to_cov_cache = calculate_cov_cache_with_old_data(
        model, tok, hparams, force_recompute=True
    )
    weights = get_weights(model, hparams, bias=True)
    current_weights_cpu = cache_weights_to_cpu(weights)
    return current_weights_cpu, layer_to_cov_cache, True


def calculate_old_loss(model, tok, hparams):
    _ensure_crispedit_defaults(hparams)
    if getattr(hparams, "disable_old_loss_check", False):
        return {}
    with torch.no_grad():
        old_task_loss = calculate_cache_loss(
            model,
            tok,
            hparams.mom2_dataset,
            sample_size=100,
        )
    return {"Task 1 Loss": old_task_loss}


def calculate_old_edit_loss(txt_chunks, tgt_chunks, model, tok):
    if len(txt_chunks) == 0:
        return {}

    mets = {}
    with torch.no_grad():
        for i, (txt, tgt) in enumerate(zip(txt_chunks, tgt_chunks)):
            request_loss = calculate_request_loss(model, tok, txt, tgt, sample_size=10)
            mets.update({f"OLD_EDIT_LOSS/Old Edit Loss Chunk {i}": request_loss})
    avg_loss = sum(mets.values()) / len(mets)
    mets.update({"Task 2 Loss": avg_loss})
    return mets


def _valid_cov_caches(layer_to_cov_caches: Optional[List[Dict[str, Dict]]]) -> List[Dict[str, Dict]]:
    if not layer_to_cov_caches:
        return []
    return [cache for cache in layer_to_cov_caches if cache]


def build_optimizer_with_cov_caches(
    model,
    hparams,
    layer_to_cov_caches: List[Dict[str, Dict]],
    opt=None,
):
    _ensure_crispedit_defaults(hparams)
    if getattr(hparams, "no_crisp", False) and opt is not None:
        return opt

    weights = get_weights(model, hparams, bias=True)
    weight_params = [v for _, v in weights.items()]

    if getattr(hparams, "no_crisp", False):
        return torch.optim.Adam(
            weight_params,
            lr=hparams.lr,
            weight_decay=hparams.weight_decay,
        )

    use_second_projection = _use_second_projection(hparams)
    valid_caches = _valid_cov_caches(layer_to_cov_caches)

    primary_projection_cache = None

    primary_projection_cache = _load_projection_cache_map_from_hparams(
        model,
        hparams,
        [
            "projection_cache_path",
            "base_projection_cache_path",
            "primary_projection_cache_path",
        ],
    )

    if valid_caches and primary_projection_cache is None:
        if use_second_projection and len(valid_caches) > 1:
            primary_cov_cache = combine_layer_to_cov_caches([valid_caches[0]])
            additional_cov_caches = valid_caches[1:]
        else:
            primary_cov_cache = combine_layer_to_cov_caches(valid_caches)
            additional_cov_caches = []

        primary_projection_cache = calculate_projection_caches_from_cov_caches(
            model, hparams, primary_cov_cache
        )
    elif use_second_projection and len(valid_caches) > 1:
        additional_cov_caches = valid_caches[1:]
    else:
        additional_cov_caches = []

    task_projection_cache = _load_projection_cache_map_from_hparams(
        model,
        hparams,
        [
            "task_projection_cache_path",
            "additional_projection_cache_path",
            "second_projection_cache_path",
        ],
    )
    if task_projection_cache is not None:
        primary_projection_cache = _merge_projection_cache_maps(
            primary_projection_cache,
            _prefix_task_projection_cache(task_projection_cache),
        )
        use_second_projection = True

    explicit_additional_cache = _load_additional_cov_cache_from_hparams(hparams)
    if explicit_additional_cache is not None:
        additional_cov_caches.append(explicit_additional_cache)
        use_second_projection = True

    if use_second_projection and additional_cov_caches:
        combined_additional_cov_cache = combine_layer_to_cov_caches(additional_cov_caches)
        second_energy_threshold = getattr(
            hparams,
            "second_energy_threshold",
            getattr(hparams, "additional_energy_threshold", None),
        )
        task_from_cov_cache = calculate_projection_caches_from_cov_caches(
            model,
            hparams,
            combined_additional_cov_cache,
            energy_threshold=second_energy_threshold,
        )
        primary_projection_cache = _merge_projection_cache_maps(
            primary_projection_cache,
            _prefix_task_projection_cache(task_from_cov_cache),
        )

    if opt is not None:
        opt.reset_cache(primary_projection_cache)
        if hasattr(opt, "reset_additional_cache"):
            opt.reset_additional_cache(None)
        for group in opt.param_groups:
            group["use_second_projection"] = use_second_projection
        return opt

    return ProjectedAdam(
        weight_params,
        projection_cache_map=primary_projection_cache,
        additional_projection_cache_map=None,
        use_second_projection=use_second_projection,
        newton_damping=getattr(hparams, "newton_damping", 1e-3),
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )


def combine_layer_to_cov_caches(
    layer_to_cov_caches: List[Dict[str, Dict]],
    normalize_trace_with_first=False,
) -> Dict[str, Dict]:
    layer_to_cov_caches = _valid_cov_caches(layer_to_cov_caches)
    if len(layer_to_cov_caches) == 0:
        return {}
    if len(layer_to_cov_caches) == 1:
        return layer_to_cov_caches[0]

    combined_layer_to_cov_caches = {}
    for layer_name in layer_to_cov_caches[0].keys():
        A_list = [layer_to_cov[layer_name]["A"] for layer_to_cov in layer_to_cov_caches]
        B_list = [layer_to_cov[layer_name]["B"] for layer_to_cov in layer_to_cov_caches]
        num_samples_list = [
            max(int(layer_to_cov[layer_name].get("num_samples", 0)), 1)
            for layer_to_cov in layer_to_cov_caches
        ]
        total_samples = sum(num_samples_list)

        combined_A = sum(
            A * num_sample for A, num_sample in zip(A_list, num_samples_list)
        ) / total_samples
        combined_B = sum(
            B * num_sample for B, num_sample in zip(B_list, num_samples_list)
        ) / total_samples

        combined_layer_to_cov_caches[layer_name] = {
            "A": combined_A,
            "B": combined_B,
            "num_samples": total_samples,
        }
        task_caches = [
            layer_to_cov[layer_name]
            for layer_to_cov in layer_to_cov_caches
            if "task_A" in layer_to_cov[layer_name] and "task_B" in layer_to_cov[layer_name]
        ]
        if task_caches:
            task_A_list = [cache["task_A"] for cache in task_caches]
            task_B_list = [cache["task_B"] for cache in task_caches]
            task_num_samples_list = [
                max(int(cache.get("task_num_samples", cache.get("num_samples", 0))), 1)
                for cache in task_caches
            ]
            task_total_samples = sum(task_num_samples_list)
            combined_layer_to_cov_caches[layer_name].update(
                {
                    "task_A": sum(
                        A * num_sample
                        for A, num_sample in zip(task_A_list, task_num_samples_list)
                    ) / task_total_samples,
                    "task_B": sum(
                        B * num_sample
                        for B, num_sample in zip(task_B_list, task_num_samples_list)
                    ) / task_total_samples,
                    "task_num_samples": task_total_samples,
                }
            )
    print(f"Combined samples {num_samples_list}")
    return combined_layer_to_cov_caches


def _find_weight_for_layer(weights: Dict[str, torch.Tensor], layer_name: str):
    if layer_name in weights:
        return weights[layer_name]

    clean = layer_name[:-len(".weight")] if layer_name.endswith(".weight") else layer_name
    for weight_name, weight in weights.items():
        if layer_name in weight_name or clean in weight_name:
            return weight
    raise KeyError(f"Could not find trainable weight for layer {layer_name}")


def calculate_projection_caches_from_cov_caches(
    model,
    hparams,
    layer_to_cov_caches,
    energy_threshold=None,
):
    weight_to_projection_cache = {}
    weights = get_weights(model, hparams, bias=False)
    device = _model_device(model)
    energy_threshold = hparams.energy_threshold if energy_threshold is None else energy_threshold

    for layer_name, cov_cache in layer_to_cov_caches.items():
        A = cov_cache["A"].to(device=device, dtype=torch.float32)
        B = cov_cache["B"].to(device=device, dtype=torch.float32)

        if not _is_llama_or_phi(hparams.model_name):
            A, B = B, A

        projection_cache = calculate_projection_cache_with_kfac(
            A, B, energy_threshold=energy_threshold
        )
        if "task_A" in cov_cache and "task_B" in cov_cache:
            task_A = cov_cache["task_A"].to(device=device, dtype=torch.float32)
            task_B = cov_cache["task_B"].to(device=device, dtype=torch.float32)
            if not _is_llama_or_phi(hparams.model_name):
                task_A, task_B = task_B, task_A
            task_projection_cache = calculate_projection_cache_with_kfac(
                task_A, task_B, energy_threshold=energy_threshold
            )
            projection_cache.update(
                {
                    f"task_{key}": value
                    for key, value in task_projection_cache.items()
                    if key in ("mask_a", "mask_b", "eig_a", "eig_b")
                }
            )
            del task_A, task_B
        projection_cache["layer_name"] = layer_name
        weight_to_projection_cache[_find_weight_for_layer(weights, layer_name)] = projection_cache

        del A, B
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return weight_to_projection_cache


def get_weights_to_projection_cache(model, opt, hparams):
    weights_to_projection_cache = opt.param_groups[0].get("projection_cache_map", {})
    weights = get_weights(model, hparams, bias=False)
    layers = _layer_names(hparams)
    layer_to_projection_cache = {}
    for layer in layers:
        weight = _find_weight_for_layer(weights, layer)
        if weight in weights_to_projection_cache:
            layer_to_projection_cache[layer] = weights_to_projection_cache[weight]
    return layer_to_projection_cache


def wrap_model_with_lora_and_return_opt(model, hparams):
    if hparams.lora_type == "lora":
        lora_config = LoraConfig
    elif hparams.lora_type == "adalora":
        lora_config = AdaLoraConfig
    else:
        raise ValueError(f"Unsupported lora_type: {hparams.lora_type}")

    peft_config = lora_config(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=hparams.lora_rank,
        lora_alpha=hparams.lora_alpha,
        lora_dropout=hparams.lora_dropout,
        layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,
        target_modules=hparams.target_modules,
    )
    peft_model = get_peft_model(model, peft_config)
    opt = torch.optim.Adam(
        peft_model.parameters(),
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    return peft_model, opt


def update_model_and_tokenizer_with_appropriate_padding_token(model, tokenizer, hparams):
    if "Qwen" in hparams.model_name:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer
