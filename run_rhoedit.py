import random
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()
from utils import (
    print_time, 
    prepare_requests_from_data_type, 
    save_model_and_tokenizer, 
)

HF_CACHE_DIR = os.getenv("HF_CACHE_DIR")
os.environ["HF_DATASETS_CACHE"] = os.getenv("HF_DATASETS_DIR")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import torch
#zwz
from transformers import AutoConfig,AutoModelForCausalLM, AutoTokenizer

from easyeditor.models.crispedit.utils import update_model_and_tokenizer_with_appropriate_padding_token
from easyeditor.models.crispedit.CrispEdit_hparams import CrispEditHyperParams
from crispedit import *

from easyeditor.models.rhoedit import execute_sft_sgd , execute_sft_adam,execute_sft_adam_sequential
from easyeditor.models.rhoedit import SGDHyperParams,AdamHyperParams

from easyeditor.tools import ExperimentTracker


SEED = 69
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True


def get_arguments():
    parser = argparse.ArgumentParser()
    # 基本信息
    parser.add_argument('--model', required=True, type=str,
                        help='Model name or path, e.g. meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--data_type', required=True, type=str, default='zsre',
                        choices=['zsre', 'zsre10k', 'counterfact', 'wiki',
                                 'safeedit_train', 'safeedit_test'])

    parser.add_argument('--alg_name', required=True, type=str, default='lora',
                        choices=['crispedit','rhoedit_adam','rhoedit_sgd'])
    parser.add_argument('--cache_sample_num', type=int, default=10000,
                        help='Number of samples to use for caching projection matrices.')
    parser.add_argument('--edit_sample_num', type=int, default=3000,
                        help='Number of samples to use for calculating old loss during editing.')


    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for fine-tuning.')

    # Sequential
    # 每个阶段编辑的数量
    parser.add_argument('--num_edits', type=int, default=100,
                        help='Sequential edit batch size.')
    # 顺序编辑开关
    parser.add_argument('--sequential_edit', action='store_true',
                        help='Whether to use sequential editing. Default is False.')
    
    # wandb/swanlab
    parser.add_argument('--wandb_project', type=str, default='CrispLoRA',
                        help='project name.')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging.')
    parser.add_argument('--plat_name', type=str, default='swanlab',
                       choices=['swanlab','wandb','none'])

    # 模型编辑一段时间后，最初计算的 K-FAC 投影矩阵可能不再适合当前模型
    # 相对参数变化超过约 25% 时触发
    parser.add_argument('--recalculate_cache', action='store_true',
                        help='Whether to recalculate the projection caches. Default is False.')
    parser.add_argument('--recalculate_weight_threshold', type=float, default=0.25,
                        help='Threshold for recalculating weight projection caches. [0.0-1.0]')
    # 顺序编辑时如何保护历史编辑
    parser.add_argument('--edit_cache_style', type=str, default='mix',
                        choices=['sequential', 'mix', 'disable'],
                        help='Cache style during sequential editing.')
                        
    # 是否使用投影
    parser.add_argument('--no_crisp', action='store_true',
                        help='Disable CrispEdit optimization (plain FT).')

    # 是否监控原始任务损失                    
    parser.add_argument('--disable_old_loss_check', action='store_true',
                        help='Disable old loss check to speed up sequential editing.')


    parser.add_argument('--perform_lora', action='store_true',
                        help='Use CrispEdit built-in LoRA mode (execute_ft_lora).')
    parser.add_argument('--lora_rank', type=int, default=32,
                        help='LoRA rank.')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha.')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                        help='LoRA dropout.')
    parser.add_argument('--lora_type', type=str, default='lora',
                        choices=['lora', 'adalora'],
                        help='Type of LoRA to use.')
    parser.add_argument('--target_modules', type=list,
                        default=["q_proj", "v_proj"],
                        help='Target modules for LoRA adaptation.')

    # my-method使用
    parser.add_argument('--newton_damping',type=float, default=1e-5)
    parser.add_argument('--soft_lambda',type=float, default=1.0)
    # --FT学习率
    parser.add_argument('--lr',type=float, default=1e-3)

    args = parser.parse_args()
    return args

def get_hparams(args):
    # 暂时拆分开进行判断，后续需要合并成一个
    if args.alg_name == "rhoedit_sgd":
        print("[run_sgd] 加载 RhoEdit SGD 配置")
        hparams = SGDHyperParams.from_hparams(f"./hparams/RhoEdit/{args.model}")

    elif args.alg_name == "rhoedit_adam":
        print("[run_adam] 加载 RhoEdit Adam 配置")
        hparams = AdamHyperParams.from_hparams(f"./hparams/RhoEdit/{args.model}")


    hparams.batch_size = args.batch_size
    hparams.mom2_n_samples = args.cache_sample_num
    hparams.task_mom2_n_samples = args.edit_sample_num
    hparams.recalculate_cache = args.recalculate_cache
    hparams.recalculate_weight_threshold = args.recalculate_weight_threshold
    hparams.edit_cache_style = args.edit_cache_style
    hparams.no_crisp = args.no_crisp
    hparams.disable_old_loss_check = args.disable_old_loss_check
    hparams.perform_lora = args.perform_lora

    hparams.lr = args.lr
    hparams.newton_damping = args.newton_damping
    hparams.soft_lambda = args.soft_lambda
    assert not (not args.no_crisp and args.perform_lora), \
        "We don't currently support using CrispEdit and LoRA together. " \
        "Please set --no_crisp if you want to use LoRA."
    if hparams.perform_lora and args.sequential_edit:
        print("Warning: We suggest using edit.py for LoRA-based sequential editing "
              "instead of this one.")

    if hparams.perform_lora:
        hparams.lora_rank = args.lora_rank
        hparams.lora_alpha = args.lora_alpha
        hparams.lora_dropout = args.lora_dropout
        hparams.lora_type = args.lora_type
        hparams.target_modules = args.target_modules

    if args.sequential_edit:
        assert args.num_edits >= args.batch_size, \
            "Makes no sense to have a batch_size bigger than number of edits..."
        hparams.num_edits = args.num_edits

    return hparams

def calculate_model_name(args, hparams):
    if args.perform_lora:
        name = f"{args.model}_LoRA_FT_{args.data_type}"
    elif args.no_crisp:
        name = f"{args.model}_FT_{args.data_type}"
    elif args.alg_name == "rhoedit_sgd" or args.alg_name == "rhoedit_adam":
        name = (f"{args.model}_{args.alg_name}_{args.data_type}"
                        f"_{hparams.newton_damping}_{hparams.soft_lambda}_{hparams.lr}")
    else:
        name = (f"{args.model}_{args.alg_name}_{args.data_type}"
                f"_{args.energy_threshold}_{hparams.mom2_n_samples}_{hparams.lr}")

    if args.sequential_edit:
        name += f"_sequential_{args.num_edits}"
    
    if hparams.recalculate_cache:
        name += f"_recalc_cache_{args.recalculate_weight_threshold}_edit_sample_{hparams.edit_sample_num}"
    if args.sequential_edit:
        name += f"_edit_cache_{args.edit_cache_style}"

    return name.replace('.', '_')

if __name__ == "__main__":
    args = get_arguments()
    requests = prepare_requests_from_data_type(args.data_type)
    requests = setup_requests_for_safeedit(requests)
    hparams = get_hparams(args)

    
    save_model_name = calculate_model_name(args, hparams)
    print(f"Model will be saved to BASE_DIR/{save_model_name}")

    ExperimentTracker.init(project=args.wandb_project, name=save_model_name, config=vars(hparams),
                            tracker_type=args.plat_name,mode = not args.no_wandb)

    MODEL_NAME = hparams.model_name
    print(f"[0] Load model ......")

    if os.path.exists(HF_CACHE_DIR+MODEL_NAME):
        MODEL_NAME=HF_CACHE_DIR+MODEL_NAME
    print(f" Load model path as:{MODEL_NAME}")
    '''
    #zwz需要保留的
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map='auto',  
                                    local_files_only=True)
    '''
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    device_map = {
        "model.embed_tokens": 0,
        "model.rotary_emb": 0,
    }

    for layer in range(config.num_hidden_layers):
        device_map[f"model.layers.{layer}"] = 0 if  15 <= layer <= 19 else 1

    device_map["model.norm"] = 1
    device_map["lm_head"] = 1

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map=device_map,
        local_files_only=True,
    )
    '''

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=HF_CACHE_DIR,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=HF_CACHE_DIR,
        device_map="auto",
        local_files_only=True,
    )
    '''
    # set appropriate padding token
    model, tokenizer = update_model_and_tokenizer_with_appropriate_padding_token(model, tokenizer, hparams)
    
    
    print_time("Begin FT Time")
    if args.alg_name == "rhoedit_sgd":
         edited_model = execute_sft_sgd(model, tokenizer, requests, hparams)
    elif args.alg_name == "rhoedit_adam":
         edited_model = execute_sft_adam(model, tokenizer, requests, hparams)
    elif args.sequential_edit:
        edited_model = execute_sft_adam_sequential(model, tokenizer, requests, hparams)

        
    print_time("End FT Time")
    save_model_and_tokenizer(edited_model, tokenizer, save_model_name)