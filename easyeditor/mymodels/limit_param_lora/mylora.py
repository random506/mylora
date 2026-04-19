      
import torch
from peft.tuners.lora.layer import LoraLayer, LoraVariant


class CurvatureLora(LoraVariant):
    """LoRA variant that applies bilateral low-curvature projection.

    The forward pass computes:
        h = (I - Ū_out Ū_out^T) B A (I - Ū_in Ū_in^T) x

    where Ū_in and Ū_out are frozen high-curvature eigenvector bases stored
    as buffers on the LoRA module, and A, B are the standard learnable LoRA
    matrices.

    The merged delta weight is:
        ΔW = (I - Ū_out Ū_out^T) @ B @ A @ (I - Ū_in Ū_in^T) * scaling
    """

    @staticmethod
    def init(module: LoraLayer, adapter_name: str, **kwargs) -> None:
        base_weight = (
            module.base_layer.weight if hasattr(module, "base_layer") else module.weight
        )
        module.register_buffer(
            f"U_in_bar_{adapter_name}",
            base_weight.new_zeros((module.in_features, 0), dtype=base_weight.dtype),
        )
        module.register_buffer(
            f"U_out_bar_{adapter_name}",
            base_weight.new_zeros((module.out_features, 0), dtype=base_weight.dtype),
        )

    @staticmethod
    def forward(
        module: LoraLayer,
        active_adapter: str,
        x: torch.Tensor,
        result: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        U_in_bar = getattr(module, f"U_in_bar_{active_adapter}")
        U_out_bar = getattr(module, f"U_out_bar_{active_adapter}")
        lora_A = module.lora_A[active_adapter]
        lora_B = module.lora_B[active_adapter]
        dropout = module.lora_dropout[active_adapter]
        scaling = module.scaling[active_adapter]

        # Cast eigenvectors to match input dtype
        U_in_bar = U_in_bar.to(dtype=x.dtype)
        U_out_bar = U_out_bar.to(dtype=x.dtype)

        # Input projection: (I - Ū_in Ū_in^T) x
        x_proj = x - (x @ U_in_bar) @ U_in_bar.T

        # Standard LoRA: B(A(dropout(x_proj)))
        h = lora_B(lora_A(dropout(x_proj)))

        # Output projection: (I - Ū_out Ū_out^T) h
        h_proj = h - (h @ U_out_bar) @ U_out_bar.T

        return result + h_proj * scaling

    @staticmethod
    def _compute_delta_weight(module: LoraLayer, active_adapter: str) -> torch.Tensor:
        """Compute (I - Ū_out Ū_out^T) @ B @ A @ (I - Ū_in Ū_in^T) * scaling."""
        U_in_bar = getattr(module, f"U_in_bar_{active_adapter}")
        U_out_bar = getattr(module, f"U_out_bar_{active_adapter}")
        weight_A = module.lora_A[active_adapter].weight
        weight_B = module.lora_B[active_adapter].weight
        scaling = module.scaling[active_adapter]

        device = weight_B.device
        dtype = weight_B.dtype

        cast_to_fp32 = device.type == "cpu" and (
            dtype == torch.float16 or dtype == torch.bfloat16
        )
        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()
            U_in_bar = U_in_bar.float()
            U_out_bar = U_out_bar.float()
        else:
            U_in_bar = U_in_bar.to(dtype=dtype)
            U_out_bar = U_out_bar.to(dtype=dtype)

        # BA: (d_out, d_in)
        BA = weight_B @ weight_A

        # Left projection: (I - Ū_out Ū_out^T) @ BA
        P_out_BA = BA - U_out_bar @ (U_out_bar.T @ BA)

        # Right projection: P_out_BA @ (I - Ū_in Ū_in^T)
        delta = P_out_BA - (P_out_BA @ U_in_bar) @ U_in_bar.T

        delta = delta * scaling

        if cast_to_fp32:
            delta = delta.to(dtype=dtype)

        return delta

    @staticmethod
    def merge_safe(
        module: LoraLayer, active_adapter: str, orig_weight: torch.Tensor
    ) -> torch.Tensor:
        delta = CurvatureLora._compute_delta_weight(module, active_adapter)
        return orig_weight + delta.to(orig_weight.dtype)

    @staticmethod
    def merge_unsafe(
        module: LoraLayer, active_adapter: str, orig_weight: torch.Tensor
    ) -> None:
        delta = CurvatureLora._compute_delta_weight(module, active_adapter)
        orig_weight.data += delta.to(orig_weight.dtype)

    @staticmethod
    def unmerge(
        module: LoraLayer, active_adapter: str, orig_weight: torch.Tensor
    ) -> torch.Tensor:
        delta = CurvatureLora._compute_delta_weight(module, active_adapter)
        return orig_weight - delta.to(orig_weight.dtype)

    