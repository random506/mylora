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
from transformers import AutoModelForCausalLM, AutoTokenizer

from easyeditor.models.crispedit.utils import update_model_and_tokenizer_with_appropriate_padding_token
from easyeditor.models.crispedit.CrispEdit_hparams import CrispEditHyperParams
from crispedit import *
from easyeditor.models.crispedit.utils_sgd import execute_ft_sgd
from easyeditor.models.crispedit.CrispEdit_hparams_sgd import (
    CrispEditHyperParams as CrispEditSGDHyperParams,
)

from easyeditor.mymodels.tools import ExperimentTracker
from easyeditor.mymodels.hparams import CrispLoRAHyperParams
from easyeditor.mymodels.crispedit_param import (
        CrispEditParamHyperParams,
        execute_crispedit_param
    )
from easyeditor.mymodels import  MyLoRAHyperParams

from easyeditor.models.ft import FTHyperParams

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
    #new
    parser.add_argument('--alg_name', required=True, type=str, default='lora',
                        choices=['crispedit',"myedit","lora",'mylora','FT','myeditpro','sgd_edit'])# 最后一个算法将优化器改为继承了sgd
    parser.add_argument('--cache_sample_num', type=int, default=10000,
                        help='Number of samples to use for caching projection matrices.')
    parser.add_argument('--edit_sample_num', type=int, default=1000,
                        help='Number of samples to use for calculating old loss during editing.')
    parser.add_argument('--energy_threshold', type=float, default=0.7,
                        help='Energy threshold for projection matrix computation.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for fine-tuning.')

    # Sequential
    parser.add_argument('--num_edits', type=int, default=100,
                        help='Sequential edit batch size.')
    parser.add_argument('--sequential_edit', action='store_true',
                        help='Whether to use sequential editing. Default is False.')
    
    # wandb/swanlab
    parser.add_argument('--wandb_project', type=str, default='CrispLoRA',
                        help='project name.')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging.')
    parser.add_argument('--plat_name', type=str, default='swanlab',
                       choices=['swanlab','wandb','none'])

    # 是否需要重新计算投影矩阵？
    parser.add_argument('--recalculate_cache', action='store_true',
                        help='Whether to recalculate the projection caches. Default is False.')
    # 重新计算投影矩阵的阈值？
    parser.add_argument('--recalculate_weight_threshold', type=float, default=0.25,
                        help='Threshold for recalculating weight projection caches. [0.0-1.0]')
    
    parser.add_argument('--no_crisp', action='store_true',
                        help='Disable CrispEdit optimization (plain FT).')
    parser.add_argument('--disable_old_loss_check', action='store_true',
                        help='Disable old loss check to speed up sequential editing.')
    # ？
    parser.add_argument('--edit_cache_style', type=str, default='mix',
                        choices=['sequential', 'mix', 'disable'],
                        help='Cache style during sequential editing.')


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

    # -- mylora 新参数
    parser.add_argument('--projection_method_lora', type=str, default=None,
                        choices=["v2_param","v2_grad","v2_both"],
                        help="Projection onto the gradient or onto the parameters")
    # 使用lora时，可以选择渗透部分
    parser.add_argument('--use_leak',action='store_true')
    parser.add_argument('--leak_rate',type=float, default=0.2)
    parser.add_argument('--newton_damping',type=float, default=0.001)

    # --myedit  新参数
    parser.add_argument('--projection_method', type=str, default=None,
                        choices=["param","both"],
                        help="Projection onto the gradient or onto the parameters")


    # --FT学习率
    parser.add_argument('--lr',type=float, default=0.7)

    args = parser.parse_args()
    return args

def get_hparams(args):
    if args.projection_method_lora is not None:
        print(f"[run_mylora] 加载 MyLoRA 配置")
        hparams = MyLoRAHyperParams.from_hparams(f"./hparams/MyLoRA/{args.model}")
        hparams.batch_size = args.batch_size
        hparams.energy_threshold = args.energy_threshold
        hparams.mom2_n_samples = args.cache_sample_num
        # 通过命令行参数修改rank
        hparams.lora_rank = args.lora_rank
        hparams.lora_alpha = args.lora_alpha
        hparams.lr =args.lr
        if args.use_leak:
            hparams.use_leak = True
            hparams.leak_rate = args.leak_rate
        # todo:连续编辑

        return hparams
    elif args.projection_method is not None:
        print(f"[run_myedit] 加载 MyEdit 配置")
        hparams = CrispEditParamHyperParams.from_hparams(f"./hparams/MyEdit/{args.model}")
        hparams.batch_size = args.batch_size
        hparams.energy_threshold = args.energy_threshold
        hparams.mom2_n_samples = args.cache_sample_num
        hparams.projection_method = args.projection_method

        return hparams
    elif args.alg_name == "FT":
        print(f"[run_FT] 加载 FT 配置")
        hparams = FTHyperParams.from_hparams(f"./hparams/FT/{args.model}")
        hparams.batch_size = args.batch_size
        hparams.lr = args.lr

        return hparams
    elif args.alg_name == "sgd_edit":
        if args.sequential_edit:
            raise ValueError(
                "--sequential_edit is not supported with --alg_name sgd."
            )
        print("[run_sgd] 加载 CrispEdit SGD 配置")
        hparams = CrispEditSGDHyperParams.from_hparams(
            f"./hparams/CrispEdit/{args.model}"
        )
        hparams.batch_size = args.batch_size
        hparams.energy_threshold = args.energy_threshold
        hparams.mom2_n_samples = args.cache_sample_num
        hparams.lr = args.lr
        assert not (not args.no_crisp and args.perform_lora), \
            "We don't currently support using CrispEdit and LoRA together. " \
            "Please set --no_crisp if you want to use LoRA."
        if hparams.perform_lora:
            hparams.lora_rank = args.lora_rank
            hparams.lora_alpha = args.lora_alpha
            hparams.lora_dropout = args.lora_dropout
            hparams.lora_type = args.lora_type
            hparams.target_modules = args.target_modules
        return hparams
    elif args.alg_name == "myeditpro":
        print(f"[run_gen_pro] 加载 Gen Pro 配置")
        hparams = CrispEditHyperParams.from_hparams(
            f"./hparams/CrispEdit/{args.model}"
        )
        hparams.batch_size = args.batch_size
        hparams.energy_threshold = args.energy_threshold
        hparams.mom2_n_samples = args.cache_sample_num
        hparams.lr = args.lr
        # Gen Pro 专属参数（有默认值，也可从 YAML 加载）
        hparams.kl_factor = getattr(hparams, 'kl_factor', 0.01)
        hparams.nullspace_threshold = getattr(hparams, 'nullspace_threshold', 0.02)
        hparams.L2 = getattr(hparams, 'L2', 1.0)
        hparams.kl_interval = getattr(hparams, 'kl_interval', 1)
        return hparams

    print(f"[run_crispedit] 加载 CrispEdit 配置")
    hparams = CrispEditHyperParams.from_hparams(f"./hparams/CrispEdit/{args.model}")
    hparams.batch_size = args.batch_size
    hparams.energy_threshold = args.energy_threshold
    hparams.mom2_n_samples = args.cache_sample_num
    hparams.edit_n_samples = args.edit_sample_num
    hparams.recalculate_cache = args.recalculate_cache
    hparams.recalculate_weight_threshold = args.recalculate_weight_threshold
    hparams.no_crisp = args.no_crisp
    hparams.disable_old_loss_check = args.disable_old_loss_check
    hparams.edit_cache_style = args.edit_cache_style
    hparams.perform_lora = args.perform_lora
    hparams.lr = args.lr
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
    if args.projection_method_lora is not None:
        name = f"{args.model}_{args.projection_method_lora}_{hparams.lora_rank}_{args.data_type}_{args.energy_threshold}_{args.cache_sample_num}_{hparams.lr}"#_{hparams.layers[0]}_{hparams.layers[-1]}"
    elif args.projection_method is not None:
        name = f"{args.model}_{args.projection_method}_{args.data_type}_{args.energy_threshold}_{args.cache_sample_num}"
    elif args.perform_lora:
        name = f"{args.model}_LoRA_FT_{args.data_type}"
    elif args.no_crisp:
        name = f"{args.model}_FT_{args.data_type}"
    elif args.alg_name == "FT":
        name = f"{args.model}_{args.alg_name}_{args.data_type}_{hparams.lr}"
        return name.replace('.', '_')
    else:
        name = (f"{args.model}_{args.alg_name}_{args.data_type}"
                f"_{args.energy_threshold}_{hparams.mom2_n_samples}_{hparams.lr}_sgd")#_gen_19-23_1e-5_new")

    if args.sequential_edit:
        name += f"_sequential_{args.num_edits}"
    
    if hparams.recalculate_cache:
        name += f"_recalc_cache_{args.recalculate_weight_threshold}_edit_sample_{hparams.edit_n_samples}"
    if args.sequential_edit:
        name += f"_edit_cache_{getattr(hparams, 'edit_cache_style', 'mix')}"

    return name.replace('.', '_')

if __name__ == "__main__":
    '''
    所有微调操作的入口函数，由命令行参数向下传递
    
    '''
    args = get_arguments()
    requests = prepare_requests_from_data_type(args.data_type)
    requests = setup_requests_for_safeedit(requests)
    hparams = get_hparams(args)

    
    save_model_name = calculate_model_name(args, hparams)
    print(f"Model will be saved to BASE_DIR/{save_model_name}")

    ExperimentTracker.init(project=args.wandb_project, name=save_model_name, config=vars(hparams), tracker_type=args.plat_name)
    #wandb.init(project=args.wandb_project, name=save_model_name, config=vars(hparams), mode="disabled" if args.no_wandb else "online")

    MODEL_NAME = hparams.model_name
    if os.path.exists(HF_CACHE_DIR+MODEL_NAME):
        MODEL_NAME=HF_CACHE_DIR+MODEL_NAME
    print(f"加载模型路径为：{MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,local_files_only=True)
    # warning
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map='auto',  
                                    local_files_only=True)
    device = model.device
    print(f"use gpu id:{device}")

    # set appropriate padding token
    model, tokenizer = update_model_and_tokenizer_with_appropriate_padding_token(model, tokenizer, hparams)
    
    
    print_time("Begin FT Time")
    print(f"超参数列表为：{hparams}")

    if args.alg_name == "sgd_edit":
        print("[1]进行 CrispEdit SGD 方法微调......")
        edited_model = execute_ft_sgd(
            model,
            tokenizer,
            requests,
            hparams
        )
    elif args.sequential_edit:
        print("[1]Crispedit微调的序列模式")
        edited_model = execute_ft_sequential(model, tokenizer, requests, hparams)
    elif args.projection_method_lora is not None:
        print("[1]针对lora进行投影......")
        if args.projection_method_lora == "v2_param":
            edited_model = execute_ft_param_lora(model, tokenizer, requests, hparams)
        elif args.projection_method_lora == "v2_grad":
            edited_model = execute_ft_grad_lora(model, tokenizer, requests, hparams)
        elif args.projection_method_lora == "v2_lora":
            edited_model = execute_ft_lora(model, tokenizer, requests, hparams)
    elif args.projection_method is not None:
        print("[1]针对矩阵进行投影......")
        if args.projection_method == "param":
            edited_model = execute_crispedit_param(model, tokenizer, requests, hparams)
    elif args.alg_name == "FT":
        print("[1]进行全量微调......")
        edited_model = execute_finetune(model, tokenizer, requests, hparams)
    elif args.alg_name == "myeditpro":
         edited_model = execute_ft_pro(model, tokenizer, requests, hparams)
    else:
        print("[1]进行Crispedit方法微调......")
        edited_model = execute_ft(model, tokenizer, requests, hparams)
    print_time("End FT Time")


    save_model_and_tokenizer(edited_model, tokenizer, save_model_name)