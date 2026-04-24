"""
CurvatureLora.py  —  crispedit_both
=====================================
复用自 mymodels/limit_lora/LeakyCurvatureLora.py。

LeakyCurvatureLora（参数侧约束）：
  - 不是彻底消除高曲率方向，而是按可学习的 leak_rate 比例保留。
  - leak_rate 是每层独立的可训练参数，初始值 sigmoid(-4) ≈ 0.018，
    被限制在 [0, 0.1]（最多 10% 的高曲率信息透过）。
  - 前向：
      x_proj  = x  - (1 - leak) * (x  @ U_in_bar ) @ U_in_bar .T
      h_proj  = h  - (1 - leak) * (h  @ U_out_bar) @ U_out_bar.T
      output  = result + h_proj * scaling

这是"Both"方案中的参数侧约束。
梯度侧约束由 GlobalAwareProjectedLoRAOptimizer 负责。
"""

import torch
import torch.nn as nn
from peft.tuners.lora.layer import LoraLayer
from peft import LoraConfig


class LeakyCurvatureLora(LoraConfig):

    @staticmethod
    def init(module: LoraLayer, adapter_name: str, **kwargs) -> None:
        base_weight = module.base_layer.weight if hasattr(module, "base_layer") else module.weight

        # 空投影基 buffer（外部调用 inject 时填入真实特征向量）
        module.register_buffer(
            f"U_in_bar_{adapter_name}",
            base_weight.new_zeros((module.in_features, 0)),
        )
        module.register_buffer(
            f"U_out_bar_{adapter_name}",
            base_weight.new_zeros((module.out_features, 0)),
        )

        # 可训练泄漏率，初始 sigmoid(-4) ≈ 0.018，最高允许 10% 高曲率信息通过
        module.register_parameter(
            f"leak_rate_{adapter_name}",
            nn.Parameter(torch.tensor([-4.0], dtype=base_weight.dtype)),
        )

    @staticmethod
    def forward(
        module: LoraLayer,
        active_adapter: str,
        x: torch.Tensor,
        result: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:

        U_in_bar  = getattr(module, f"U_in_bar_{active_adapter}") .to(dtype=x.dtype)
        U_out_bar = getattr(module, f"U_out_bar_{active_adapter}").to(dtype=x.dtype)

        raw_leak = getattr(module, f"leak_rate_{active_adapter}")
        leak = torch.sigmoid(raw_leak) * 0.1   # 限制在 [0, 0.1]

        lora_A  = module.lora_A[active_adapter]
        lora_B  = module.lora_B[active_adapter]
        dropout = module.lora_dropout[active_adapter]
        scaling = module.scaling[active_adapter]

        # 输入侧：软去除高曲率分量
        high_curv_in = (x @ U_in_bar) @ U_in_bar.T
        x_proj = x - (1.0 - leak) * high_curv_in

        h = lora_B(lora_A(dropout(x_proj)))

        # 输出侧：软去除高曲率分量
        high_curv_out = (h @ U_out_bar) @ U_out_bar.T
        h_proj = h - (1.0 - leak) * high_curv_out

        return result + h_proj * scaling

    @staticmethod
    def _compute_delta_weight(module: LoraLayer, active_adapter: str) -> torch.Tensor:
        U_in_bar  = getattr(module, f"U_in_bar_{active_adapter}")
        U_out_bar = getattr(module, f"U_out_bar_{active_adapter}")
        weight_A  = module.lora_A[active_adapter].weight
        weight_B  = module.lora_B[active_adapter].weight
        scaling   = module.scaling[active_adapter]

        leak = torch.sigmoid(getattr(module, f"leak_rate_{active_adapter}")) * 0.1

        dtype = weight_B.dtype
        U_in_bar  = U_in_bar .to(dtype)
        U_out_bar = U_out_bar.to(dtype)

        BA = weight_B @ weight_A   # (d_out, d_in)

        # P_out = I - (1 - leak) * U_out U_out^T
        P_out_BA = BA - (1.0 - leak) * U_out_bar @ (U_out_bar.T @ BA)
        # P_in  = I - (1 - leak) * U_in  U_in^T
        delta = P_out_BA - (1.0 - leak) * (P_out_BA @ U_in_bar) @ U_in_bar.T

        return delta * scaling
