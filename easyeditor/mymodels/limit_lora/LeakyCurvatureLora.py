import torch
import torch.nn as nn
from peft.tuners.lora.layer import LoraLayer
from peft import LoraConfig 

class LeakyCurvatureLora(LoraConfig):
    
    @staticmethod
    def init(module: LoraLayer, adapter_name: str, **kwargs) -> None:
        base_weight = module.base_layer.weight if hasattr(module, "base_layer") else module.weight
        
        module.register_buffer(f"U_in_bar_{adapter_name}", base_weight.new_zeros((module.in_features, 0)))
        module.register_buffer(f"U_out_bar_{adapter_name}", base_weight.new_zeros((module.out_features, 0)))
        
        # 新增：可训练的泄漏率参数，初始化为一个很小的值 (比如 sigmoid(-4) ≈ 0.018)
        # 允许高曲率信息极其微弱地流通
        module.register_parameter(
            f"leak_rate_{adapter_name}", 
            nn.Parameter(torch.tensor([-4.0], dtype=base_weight.dtype))
        )

    @staticmethod
    def forward(
        module: LoraLayer, active_adapter: str, x: torch.Tensor, result: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        
        U_in_bar = getattr(module, f"U_in_bar_{active_adapter}").to(dtype=x.dtype)
        U_out_bar = getattr(module, f"U_out_bar_{active_adapter}").to(dtype=x.dtype)
        
        # 获取动态泄漏率，限制在 [0, max_leak] 之间，防止过大破坏预训练知识
        raw_leak = getattr(module, f"leak_rate_{active_adapter}")
        leak = torch.sigmoid(raw_leak) * 0.1  # 最高只允许 10% 的高曲率信息泄漏

        lora_A = module.lora_A[active_adapter]
        lora_B = module.lora_B[active_adapter]
        dropout = module.lora_dropout[active_adapter]
        scaling = module.scaling[active_adapter]

        # 改进的输入侧投影：不是彻底减去，而是按比例保留 (leak)
        # x_proj = x - (1 - leak) * 高曲率分量
        high_curv_in = (x @ U_in_bar) @ U_in_bar.T
        x_proj = x - (1.0 - leak) * high_curv_in

        h = lora_B(lora_A(dropout(x_proj)))

        # 改进的输出侧投影
        high_curv_out = (h @ U_out_bar) @ U_out_bar.T
        h_proj = h - (1.0 - leak) * high_curv_out

        return result + h_proj * scaling
        
    @staticmethod
    def _compute_delta_weight(module: LoraLayer, active_adapter: str) -> torch.Tensor:
        # 在合并权重时，也需要将 leak 计算进去
        U_in_bar = getattr(module, f"U_in_bar_{active_adapter}")
        U_out_bar = getattr(module, f"U_out_bar_{active_adapter}")
        weight_A = module.lora_A[active_adapter].weight
        weight_B = module.lora_B[active_adapter].weight
        scaling = module.scaling[active_adapter]
        
        device = weight_B.device
        dtype = weight_B.dtype
        leak = torch.sigmoid(getattr(module, f"leak_rate_{active_adapter}").to(device=device)) * 0.1
        U_in_bar, U_out_bar = U_in_bar.to(device=device, dtype=dtype), U_out_bar.to(device=device, dtype=dtype)
        
        BA = weight_B @ weight_A
        
        # P_out = I - (1 - leak) * U_out U_out^T
        P_out_BA = BA - (1.0 - leak) * U_out_bar @ (U_out_bar.T @ BA)
        # P_in = I - (1 - leak) * U_in U_in^T
        delta = P_out_BA - (1.0 - leak) * (P_out_BA @ U_in_bar) @ U_in_bar.T
        
        return delta * scaling