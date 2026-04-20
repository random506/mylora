import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

# 这里导入你自己的 CurvatureLora 实现
from .CurvatureLora import CurvatureLora

def attach_curvature_lora_variant(peft_model, adapter_name="default"):
    """
    对 peft_model 中所有 LoRA Linear 层：
    1) 注册 CurvatureLora variant
    2) 调用 CurvatureLora.init(...) 创建 buffer
    """
    count = 0
    for name, module in peft_model.named_modules():
        # 一个很实用的判断方式：
        # 只要它具有 LoRA 常见字段，就当作 LoRA layer 处理
        if (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and hasattr(module, "in_features")
            and hasattr(module, "out_features")
        ):
            # 某些模块可能不是我们要的 adapter，先判断一下
            if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
                continue

            # 如果没有 lora_variant 字典，就跳过
            if not hasattr(module, "lora_variant"):
                print("[ERROR]: ")
                continue

            # 后挂载 CurvatureLora
            module.lora_variant[adapter_name] = CurvatureLora()

            # 初始化该 variant 所需的 buffer:
            # U_in_bar_{adapter_name}, U_out_bar_{adapter_name}
            module.lora_variant[adapter_name].init(module, adapter_name=adapter_name)
            count += 1

    print(f"[attach_curvature_lora_variant] 已挂载 {count} 个 CurvatureLora 层")


