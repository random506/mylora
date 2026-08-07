import argparse
import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("HF_ENDPOINT =", os.getenv("HF_ENDPOINT"))
print("HF_DATASETS_DIR =", os.getenv("HF_DATASETS_DIR"))

import numpy as np
import torch
from easyeditor.tools.tracker import ExperimentTracker
from lm_eval import simple_evaluate

HF_DATASETS_DIR = os.getenv("HF_DATASETS_DIR")
if HF_DATASETS_DIR:
    os.environ.setdefault("HF_DATASETS_CACHE", HF_DATASETS_DIR)
    os.environ.setdefault("HF_HUB_CACHE", os.getenv("HF_HUB_CACHE", HF_DATASETS_DIR))

SEED = 69
KNOWN_TASKS = ("ifeval", "truthfulqa_mc2", "mmlu", "gsm8k_cot", "arc_challenge")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def resolve_model_path(edited_model_dir_local):
    prefix_dir = os.getenv("HF_CACHE_DIR", "")
    if prefix_dir and not os.path.isabs(edited_model_dir_local):
        return os.path.join(prefix_dir, edited_model_dir_local)
    return edited_model_dir_local


def parse_tasks(tasks_arg):
    if not tasks_arg or tasks_arg.strip().lower() == "all":
        return list(KNOWN_TASKS)
    tasks = [task.strip() for task in tasks_arg.split(",") if task.strip()]
    if not tasks:
        raise ValueError("--tasks was provided but no valid task names were found.")
    return tasks


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edited_model_dir", required=True, type=str)
    parser.add_argument(
        "--data_type",
        required=True,
        type=str,
        default="zsre",
        choices=["zsre", "counterfact", "wiki", "safeedit_train", "safeedit_test"],
    )
    parser.add_argument("--eval_num", type=int, default=200)
    parser.add_argument("--alg_name", required=True, type=str, default="ft_edit")
    parser.add_argument("--model_name", required=True, type=str, default="gpt2-xl")
    parser.add_argument("--wandb_project", type=str, default="CrispEdit_EVAL")
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")

    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help='Comma-separated lm_eval tasks, or "all".',
    )
    parser.add_argument("--capability_batch_size", type=str, default="auto")
    parser.add_argument("--max_batch_size", type=int, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.40)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--swap_space", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--tokenizer", type=str, default=None)

    parser.add_argument("--apply_chat_template", action="store_true")
    parser.add_argument(
        "--no_apply_chat_template",
        action="store_false",
        dest="apply_chat_template",
        help="Disable chat template for non-instruct/base models.",
    )
    parser.set_defaults(apply_chat_template=True)

    return parser.parse_args()


def build_task_config(args):
    default_batch_size = args.capability_batch_size
    heavy_batch_size = 1 if default_batch_size == "auto" else default_batch_size
    cot_batch_size = 2 if default_batch_size == "auto" else default_batch_size

    task_defaults = {
        "ifeval": {"shots": 0, "batch_size": 4},
        "truthfulqa_mc2": {"shots": 0, "batch_size": default_batch_size},
        "mmlu": {"shots": 5, "batch_size": default_batch_size},
        "gsm8k_cot": {"shots": 8, "batch_size": cot_batch_size},
        "arc_challenge": {"shots": 25, "batch_size": heavy_batch_size},
    }

    tasks_with_config = {}
    for task_name in parse_tasks(args.tasks):
        tasks_with_config[task_name] = task_defaults.get(
            task_name,
            {"shots": 0, "batch_size": default_batch_size},
        )
    return tasks_with_config


def build_vllm_model_args(args, model_path):
    model_args = {
        "pretrained": model_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "max_model_len": args.max_model_len,
        "swap_space": args.swap_space,
        "seed": args.seed,
    }
    if args.tokenizer:
        model_args["tokenizer"] = args.tokenizer
    return model_args


def save_results(results, log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    result_path = Path(log_dir) / "capability.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    return result_path


def main():
    args = get_arguments()
    set_seed(args.seed)

    model_path = resolve_model_path(args.edited_model_dir)
    run_name = args.edited_model_dir.replace("/", "_").replace("\\", "_").strip("_")
    run_config = {
        "alg_name": args.alg_name,
        "model_name": args.model_name,
        "data_type": args.data_type,
        "backend": "vllm",
        "model_path": model_path,
        "tasks": args.tasks,
    }

    ExperimentTracker.init(
        project=args.wandb_project,
        name=run_name,
        config=run_config,
        mode=(not args.no_wandb),
    )


    model_args = build_vllm_model_args(args, model_path)
    print(f"Loading vLLM via lm_eval with model_args={model_args}")

    start_time = time.time()
    results = {"results": {}}
    tasks_with_config = build_task_config(args)

    for task_name, config in tasks_with_config.items():
        task_limit = args.eval_num if task_name == "mmlu" else None
        limit_label = task_limit if task_limit is not None else "full"
        print(
            f"Running {task_name} "
            f"(Shots: {config['shots']}, Batch: {config['batch_size']}, Limit: {limit_label})..."
        )
        task_results = simple_evaluate(
            model="vllm",
            model_args=model_args,
            tasks=[task_name],
            limit=task_limit,
            num_fewshot=config["shots"],
            batch_size=config["batch_size"],
            max_batch_size=args.max_batch_size,
            apply_chat_template=args.apply_chat_template,
            fewshot_as_multiturn=args.apply_chat_template,
            random_seed=args.seed,
            numpy_random_seed=args.seed,
            torch_random_seed=args.seed,
            fewshot_random_seed=args.seed,
        )

        if "results" not in task_results:
            raise ValueError(f"No results found for task {task_name}")

        results["results"].update(task_results["results"])
        ExperimentTracker.log(task_results["results"])

    log_dir = Path("logs") / run_name
    result_path = save_results(results, log_dir)
    elapsed = time.time() - start_time
    print(f"Raw results saved to: {result_path}")
    print(f"End Capability Eval Time: {elapsed:.2f}s")

    ExperimentTracker.finish()


if __name__ == "__main__":
    main()
