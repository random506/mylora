import os
import json
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL=os.getenv("BASE_URL")
MODEL=os.getenv("MODEL")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse
from utils import print_time, prepare_requests_from_data_type
from easyeditor.editors.utils import summary_metrics
from transformers import AutoTokenizer
from vllm import LLM
import numpy as np
from easyeditor.evaluate.evaluate import compute_edit_quality, compute_edit_quality_safety
from easyeditor.evaluate.evaluate_utils import resolve_pending_llm_judges
import random
import torch
from tqdm import tqdm
from easyeditor.tools import ExperimentTracker
from easyeditor.util import HyperParams


SEED = 69420
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

def get_model_and_tokenizer_from_dir(edited_model_dir_local):
    # 替换为vllm库加载模型
    PREFIX_DIR = os.getenv("HF_CACHE_DIR")
    edited_model_dir = PREFIX_DIR + edited_model_dir_local
    print(f"[vLLM] Loading model from: {edited_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        edited_model_dir,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = LLM(
        model=edited_model_dir,
        tokenizer=edited_model_dir,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    )
    return model, tokenizer

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edited_model_dir', required=True, type=str, default=None, help='Path to edited model for evaluation.')
    parser.add_argument('--data_type', required=True, type=str, default='zsre', choices=['zsre', 'counterfact', 'multi_counterfact', 'wiki', 'safeedit_train', 'safeedit_test'])
    parser.add_argument('--eval_num', required=False, type=int, default=3000, help='Number of evaluation instances to use. Default uses all.')
    parser.add_argument('--max_length', required=False, type=int, default=40, help='Maximum length of the generated sequences.')
    parser.add_argument('--context_type', required=True, type=str, default='qa_inst', choices=['qa_inst', 'chat_temp', 'no_context'], help='Type of context to use for evaluation.')
    parser.add_argument('--alg_name', required=True, type=str, default='ft_edit', help='Name of the editing algorithm used.')
    parser.add_argument('--model_name', required=True, type=str, default='gpt2-xl', help='Name of the base model used.')
    parser.add_argument('--evaluation_criteria', required=True, type=str, default='exact_match', choices=['exact_match', 'llm_judge'], help='Evaluation criteria to use.')  
    parser.add_argument('--wandb_project', type=str, default='CrispEdit_EVAL', help='WandB project name.')
    parser.add_argument('--wandb_run_id', type=str, default=None, help='WandB run ID for resuming runs.')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging.')
    parser.add_argument('--judge_batch_size', required=False, type=int, default=16, help='Number of generated answers to judge per batch.')
    args = parser.parse_args()
    return args

def build_hparams_from_args(args):
    hparams = HyperParams()
    hparams.alg_name = args.alg_name
    hparams.context_type = args.context_type
    hparams.max_length = args.max_length
    hparams.api_key = API_KEY
    hparams.base_url = BASE_URL
    hparams.model = MODEL
    hparams.evaluation_type = "WILD"
    hparams.model_name = args.model_name
    hparams.evaluation_criteria = args.evaluation_criteria
    hparams.data_type = args.data_type
    return hparams

if __name__ == "__main__":
    args = get_arguments()
    hparams = build_hparams_from_args(args)
    requests = prepare_requests_from_data_type(args.data_type)
    model, tokenizer = get_model_and_tokenizer_from_dir(args.edited_model_dir)
    # device expects the device number only
    device = 0

    run_name = args.edited_model_dir + f"_eval_{args.evaluation_criteria}_{args.context_type}"
    ExperimentTracker.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(hparams),
        tracker_type="wandb",
        mode=not args.no_wandb,
    )

    # before evaluation, always make sure tokenizer padding side is correct


    print_time("Begin Post Edit Eval Time")
    requests = random.sample(requests, len(requests))
    if args.eval_num is not None:
        requests = requests[:args.eval_num]

    all_metrics = []
    log_dir = f"./logs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)

    print("="*50)
    print("safe" not in args.data_type)
    edit_eval_method = compute_edit_quality if "safe" not in args.data_type else compute_edit_quality_safety
    for i, request in enumerate(tqdm(requests)):
        metrics = {
            'case_id': i,
            "requested_rewrite": request,
            "pre": {},
            "post": edit_eval_method(model, hparams.model_name, hparams, tokenizer, request, device)
        }
        all_metrics.append(metrics)
        with open(os.path.join(log_dir, "results_pending_judge.json"), "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=4)

    if args.evaluation_criteria == "llm_judge":
        all_metrics = resolve_pending_llm_judges(
            all_metrics,
            hparams.api_key,
            batch_size=args.judge_batch_size,
        )

    summary_metrics(all_metrics, log_dir)

    print_time("End Post Edit Eval Time") 
    ExperimentTracker.finish()
