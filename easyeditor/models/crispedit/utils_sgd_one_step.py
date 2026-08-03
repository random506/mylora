import random
import gc
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
from dotenv import load_dotenv
from peft import AdaLoraConfig, LoraConfig, TaskType, get_peft_model
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
from easyeditor.mymodels.tools import ExperimentTracker

# Standalone SGD variant generated from utils.py_gen_pro.bak.
from .projected_adam_sgd_one_step import ProjectedSGD
from ..rome.layer_stats import (
    calculate_cache_loss,
    calculate_request_loss,
    layer_stats_kfac_one_pass,
    layer_stats_kfac_with_txt_tgt,
)
from .CrispEdit_hparams_sgd import CrispEditHyperParams

load_dotenv()
STATS_DIR = os.getenv("STATS_DIR")


def _is_llama_or_phi(model_name: str) -> bool:
    lower = str(model_name).lower()
    return "llama" in lower or "phi" in lower or "qwen" in lower


def _model_device(model) -> torch.device:
    return getattr(model, "device", next(model.parameters()).device)


def _layer_names(hparams) -> List[str]:
    return [hparams.rewrite_module_tmp.format(layer) for layer in hparams.layers]


def _cache_dtype_name(hparams) -> str:
    return getattr(hparams, "mom2_n_dtype", getattr(hparams, "mom2_dtype", "float32"))


def _cache_sample_size(hparams) -> int:
    return int(getattr(hparams, "mom2_n_sample", getattr(hparams, "mom2_n_samples", 10000)))


def _build_cov_cache_from_hparams(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams,
    force_recompute: bool = False,
) -> Dict[str, Dict]:
    layer_names = _layer_names(hparams)
    dtype_name = _cache_dtype_name(hparams)
    sample_size = _cache_sample_size(hparams)

    task_mom2_dataset = getattr(hparams, "task_mom2_dataset", None)
    task_sample_size = int(getattr(hparams, "task_mom2_n_samples", None) or sample_size)
    print("[CrispEdit] Computing/loading base KFAC stats.")
    stats_dict = layer_stats_kfac_one_pass(
        model=model,
        tokenizer=tok,
        layer_names=layer_names,
        stats_dir=STATS_DIR,
        ds_name=hparams.mom2_dataset,
        to_collect=["mom2"],
        sample_size=sample_size if not force_recompute else sample_size,
        precision=dtype_name,
        force_recompute=force_recompute
    )

    task_stats_dict=None
    if task_mom2_dataset is not None:
        print("[CrispEdit] Computing/loading task KFAC stats.")
        task_stats_dict = layer_stats_kfac_one_pass(
        model=model,
        tokenizer=tok,
        layer_names=layer_names,
        stats_dir=STATS_DIR,
        ds_name=task_mom2_dataset,
        to_collect=["mom2"],
        sample_size=task_sample_size if not force_recompute else task_sample_size,
        precision=dtype_name,
        force_recompute=force_recompute
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


def _to_cpu_float32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to("cpu", dtype=torch.float32).contiguous()


def calculate_projection_cache_with_kfac(A, B, energy_threshold=0.9):
    del energy_threshold
    return {
        "A": _to_cpu_float32(A),
        "B": _to_cpu_float32(B),
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
    if getattr(hparams, "no_crisp", False):
        return None
    return _build_cov_cache_from_hparams(model, tok, hparams, force_recompute)


def calculate_cov_cache_with_request(txt, tgt, model, tok, hparams):
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


def _sgd_kwargs(hparams) -> Dict:
    return {
        "lr": hparams.lr,
        "momentum": float(
            getattr(hparams, "sgd_momentum", getattr(hparams, "momentum", 0.0))
        ),
        "dampening": float(
            getattr(hparams, "sgd_dampening", getattr(hparams, "dampening", 0.0))
        ),
        "weight_decay": hparams.weight_decay,
        "nesterov": bool(
            getattr(hparams, "sgd_nesterov", getattr(hparams, "nesterov", False))
        ),
        "maximize": bool(
            getattr(hparams, "sgd_maximize", getattr(hparams, "maximize", False))
        ),
    }


def _build_plain_sgd(params, hparams):
    return torch.optim.SGD(params, **_sgd_kwargs(hparams))


def _exact_sgd_kwargs() -> Dict:
    return {
        "momentum": 0.0,
        "dampening": 0.0,
        "weight_decay": 0.0,
        "nesterov": False,
        "maximize": False,
    }


def build_optimizer_with_cov_caches(
    model,
    hparams,
    layer_to_cov_caches: List[Dict[str, Dict]],
    opt=None,
):
    if getattr(hparams, "no_crisp", False) and opt is not None:
        return opt

    weights = get_weights(model, hparams, bias=True)
    weight_params = list(weights.values())

    if getattr(hparams, "no_crisp", False):
        return _build_plain_sgd(weight_params, hparams)

    valid_caches = _valid_cov_caches(layer_to_cov_caches)
    primary_cov_cache = (
        combine_layer_to_cov_caches([valid_caches[0]]) if valid_caches else {}
    )
    primary_projection_cache = (
        calculate_projection_caches_from_cov_caches(
            model,
            hparams,
            primary_cov_cache,
        )
        if primary_cov_cache
        else {}
    )
    soft_lambda = getattr(
        hparams,
        "soft_lambda",
        getattr(hparams, "newton_lambda", getattr(hparams, "lambda_soft", 1.0)),
    )
    factor_damping = getattr(
        hparams,
        "factor_damping",
        getattr(hparams, "newton_damping", 1e-3),
    )

    if isinstance(opt, ProjectedSGD):
        for group in opt.param_groups:
            group.update(_exact_sgd_kwargs())
            group["soft_lambda"] = float(soft_lambda)
        opt.factor_damping = max(float(factor_damping), 0.0)
        opt.reset_cache(primary_projection_cache)
        return opt

    return ProjectedSGD(
        weight_params,
        projection_cache_map=primary_projection_cache,
        soft_lambda=soft_lambda,
        factor_damping=factor_damping,
        lr= getattr(hparams, "lr", 5e-4),
        **_exact_sgd_kwargs(),
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
    print(f"[CrispEdit-SGD] Combined {len(layer_to_cov_caches)} covariance caches.")
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
    # 不使用阈值来限制
    del energy_threshold
    weight_to_projection_cache = {}
    weights = get_weights(model, hparams, bias=False)
    device = _model_device(model)

    for layer_name, cov_cache in layer_to_cov_caches.items():
        A = cov_cache["A"].to(device=device, dtype=torch.float32)
        B = cov_cache["B"].to(device=device, dtype=torch.float32)

        if not _is_llama_or_phi(hparams.model_name):
            A, B = B, A

        projection_cache = calculate_projection_cache_with_kfac(
            A, B
        )
        projection_cache.update(
            {
                "cap_A": projection_cache["A"],
                "cap_B": projection_cache["B"],
            }
        )

        if "task_A" in cov_cache and "task_B" in cov_cache:
            task_A = cov_cache["task_A"].to(device=device, dtype=torch.float32)
            task_B = cov_cache["task_B"].to(device=device, dtype=torch.float32)
            task_num_samples = cov_cache.get("task_num_samples")
        else:
            print(
                "[CrispEdit-New] Skipping soft K-FAC cache for "
                f"{layer_name}: missing edit/task K-FAC factors."
            )
            del A, B
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        if not _is_llama_or_phi(hparams.model_name):
            task_A, task_B = task_B, task_A

        task_projection_cache = calculate_projection_cache_with_kfac(task_A, task_B)
        projection_cache.update(
            {
                "edit_A": task_projection_cache["A"],
                "edit_B": task_projection_cache["B"],
                "task_num_samples": task_num_samples,
            }
        )
        projection_cache["layer_name"] = layer_name
        weight_to_projection_cache[_find_weight_for_layer(weights, layer_name)] = projection_cache

        del A, B, task_A, task_B
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(
        f"[CrispEdit-SGD] Built soft K-FAC caches for "
        f"{len(weight_to_projection_cache)} weights."
    )
    return weight_to_projection_cache


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
    opt = _build_plain_sgd(peft_model.parameters(), hparams)
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


def _chunks(items, chunk_size):
    chunk = []
    for item in items:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


class _AverageMeter:
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


def _execute_one_shot_projected_sgd(
    model,
    tok,
    requests,
    hparams,
    opt: ProjectedSGD,
    device,
):
    if not requests:
        raise ValueError("The section 3.2 exact update requires at least one request.")

    configured_lr = float(getattr(hparams, "lr", 1.0))
    configured_steps = int(getattr(hparams, "num_steps", 1))
    if configured_lr != 1.0:
        print(
            f"[CrispEdit-SGD] Ignoring configured lr={configured_lr:g}; "
            "the one-shot section 3.2 update uses lr=1."
        )
    if configured_steps != 1:
        print(
            f"[CrispEdit-SGD] Ignoring configured num_steps={configured_steps}; "
            "the closed-form update is applied exactly once."
        )

    print(
        "[CrispEdit-SGD] Accumulating one full target-token mean gradient "
        "with momentum=0 and weight_decay=0."
    )
    texts = [request["prompt"] for request in requests]
    targets = [request["target_new"] for request in requests]

    opt.zero_grad(set_to_none=True)
    total_loss_sum = 0.0
    total_target_tokens = 0

    for text_batch, target_batch in zip(
        _chunks(texts, hparams.batch_size),
        _chunks(targets, hparams.batch_size),
    ):
        inputs_targets = [
            text + target
            for text, target in zip(text_batch, target_batch)
        ]
        encodings = tok(
            inputs_targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=hparams.max_length,
        ).to(device)
        labels = encodings["input_ids"].clone()
        labels[labels == tok.pad_token_id] = -100

        for index, prompt in enumerate(text_batch):
            prompt_len = len(
                tok(
                    prompt,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=hparams.max_length,
                )["input_ids"]
            )
            labels[index, :prompt_len] = -100

        # Causal-LM loss ignores labels[:, 0] after shifting logits and labels.
        target_tokens = int(labels[..., 1:].ne(-100).sum().item())
        if target_tokens == 0:
            continue

        outputs = model(**encodings, labels=labels)
        loss = outputs.loss
        if not torch.isfinite(loss).item():
            raise RuntimeError("Encountered a non-finite edit loss while accumulating gradients.")

        # Hugging Face returns the mean over valid target tokens. Reconstruct
        # each batch sum so the final division yields the global token mean.
        (loss * target_tokens).backward()
        total_loss_sum += loss.detach().item() * target_tokens
        total_target_tokens += target_tokens

    if total_target_tokens == 0:
        raise RuntimeError(
            "No target tokens survived tokenization/truncation; the exact update cannot be computed."
        )

    with torch.no_grad():
        for group in opt.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.div_(total_target_tokens)

    average_loss = total_loss_sum / total_target_tokens
    opt.step()
    ExperimentTracker.log(
        {
            "FT Loss": average_loss,
            "FT Target Tokens": total_target_tokens,
        }
    )
    print(
        f"[CrispEdit-SGD] Applied one exact soft K-FAC step: "
        f"loss={average_loss:.6f}, target_tokens={total_target_tokens}."
    )


def execute_ft_sgd(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: CrispEditHyperParams,
    **kwargs: Any,
) -> AutoModelForCausalLM:
    """Apply one exact soft K-FAC update, or run the plain-SGD fallback."""

    print("进入执行函数")
    device = model.device
    if tok.padding_side != "right":
        tok.padding_side = "right"

    requests = deepcopy(requests)
    for index, request in enumerate(requests):
        if request["target_new"] and request["target_new"][0] != " ":
            requests[index]["target_new"] = " " + request["target_new"]

    layer_to_cov_cache_old = calculate_cov_cache_with_old_data(
        model,
        tok,
        hparams,
        force_recompute=False,
    )

    if hparams.perform_lora:
        model, opt = wrap_model_with_lora_and_return_opt(model, hparams)
        current_weights_cpu = None
    else:
        opt = build_optimizer_with_cov_caches(
            model,
            hparams,
            [layer_to_cov_cache_old],
        )

        weights = get_weights(model, hparams, bias=True)
        current_weights_cpu = (
            None
            if isinstance(opt, ProjectedSGD)
            else cache_weights_to_cpu(weights)
        )
        for name, weight in model.named_parameters():
            weight.requires_grad = name in weights

    if isinstance(opt, ProjectedSGD):
        _execute_one_shot_projected_sgd(
            model,
            tok,
            requests,
            hparams,
            opt,
            device,
        )
        return model

    loss_meter = _AverageMeter()
    pbar = trange(hparams.num_steps)
    for _ in pbar:
        loss_meter.reset()
        random.shuffle(requests)
        texts = [request["prompt"] for request in requests]
        targets = [request["target_new"] for request in requests]

        for text_batch, target_batch in zip(
            _chunks(texts, hparams.batch_size),
            _chunks(targets, hparams.batch_size),
        ):
            inputs_targets = [
                text + target
                for text, target in zip(text_batch, target_batch)
            ]
            encodings = tok(
                inputs_targets,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=hparams.max_length,
            ).to(device)
            labels = encodings["input_ids"].clone()
            labels[labels == tok.pad_token_id] = -100

            for index, prompt in enumerate(text_batch):
                prompt_len = len(
                    tok(
                        prompt,
                        add_special_tokens=True,
                        truncation=True,
                        max_length=hparams.max_length,
                    )["input_ids"]
                )
                labels[index, :prompt_len] = -100

            opt.zero_grad(set_to_none=True)
            outputs = model(**encodings, labels=labels)
            loss = outputs.loss
            loss_meter.update(loss.item(), n=labels.size(0))

            if loss.item() >= 1e-2:
                loss.backward()
                opt.step()
                (
                    current_weights_cpu,
                    layer_to_cov_cache_old,
                    should_recalculate,
                ) = recalculate_cov_cache_if_weights_changed(
                    model,
                    tok,
                    hparams,
                    current_weights_cpu,
                    layer_to_cov_cache_old,
                )
                if should_recalculate:
                    opt = build_optimizer_with_cov_caches(
                        model,
                        hparams,
                        [layer_to_cov_cache_old],
                        opt=opt,
                    )

        metrics = {"FT Loss": loss_meter.avg}
        ExperimentTracker.log(metrics)
        pbar.write(f"FT Loss: {loss_meter.avg:.4f}")
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

        if loss_meter.avg < 1e-2:
            break

    if hparams.perform_lora:
        model = model.merge_and_unload()

    return model
