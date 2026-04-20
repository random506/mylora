import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

# 这里导入你自己的 CurvatureLora 实现
from .CurvatureLora import CurvatureLora


#
# =========================
# 3. 将普通 LoRA 层“后挂载”为 CurvatureLora
# =========================
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
                continue

            # 后挂载 CurvatureLora
            module.lora_variant[adapter_name] = CurvatureLora()

            # 初始化该 variant 所需的 buffer:
            # U_in_bar_{adapter_name}, U_out_bar_{adapter_name}
            module.lora_variant[adapter_name].init(module, adapter_name=adapter_name)
            count += 1

    print(f"[attach_curvature_lora_variant] 已挂载 {count} 个 CurvatureLora 层")


# =========================
# 4. 为每个 LoRA 层写入高曲率方向基
# =========================
def set_curvature_bases(
    peft_model,
    adapter_name="default",
    k_in=2,
    k_out=2,
    only_modules=None,
):
    """
    给每个 LoRA 层写入 U_in_bar / U_out_bar。
    这里用随机正交基模拟真实的 K-FAC 高曲率方向。

    参数:
    - only_modules: 若不为 None，则只给模块名中包含这些字符串的层设置
      例如 only_modules=["fc1", "fc2"]
    """
    count = 0
    for name, module in peft_model.named_modules():
        if only_modules is not None and not any(key in name for key in only_modules):
            continue

        if (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and hasattr(module, "in_features")
            and hasattr(module, "out_features")
        ):
            if adapter_name not in getattr(module, "lora_variant", {}):
                continue

            # 基础 dtype/device 通常跟 base weight 对齐
            base_weight = (
                module.base_layer.weight
                if hasattr(module, "base_layer")
                else module.weight
            )
            device = base_weight.device
            dtype = base_weight.dtype

            # 构造随机正交基，模拟“高曲率方向”
            U_in_bar = random_orthonormal_basis(
                dim=module.in_features,
                k=min(k_in, module.in_features),
                device=device,
                dtype=dtype,
            )
            U_out_bar = random_orthonormal_basis(
                dim=module.out_features,
                k=min(k_out, module.out_features),
                device=device,
                dtype=dtype,
            )

            # 直接覆盖掉 CurvatureLora.init 时注册的空 buffer
            setattr(module, f"U_in_bar_{adapter_name}", U_in_bar)
            setattr(module, f"U_out_bar_{adapter_name}", U_out_bar)

            count += 1
            print(
                f"[set_curvature_bases] {name}: "
                f"U_in_bar={tuple(U_in_bar.shape)}, "
                f"U_out_bar={tuple(U_out_bar.shape)}"
            )

    print(f"[set_curvature_bases] 已设置 {count} 个层的曲率基")



# =========================
# 6. 打印当前每层的 delta weight 形状
# =========================
@torch.no_grad()
def inspect_delta_weights(peft_model, adapter_name="default"):
    for name, module in peft_model.named_modules():
        if (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and hasattr(module, "lora_variant")
            and adapter_name in module.lora_variant
        ):
            delta = CurvatureLora._compute_delta_weight(module, adapter_name)
            print(f"[delta] {name}: delta.shape = {tuple(delta.shape)}")
            print(delta)
