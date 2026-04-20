import torch
from peft.tuners.lora.layer import LoraLayer
from peft import LoraConfig 

class CurvatureLora(LoraConfig):
    """
    CurvatureLora：一种 LoRA 的变体。

    它的核心数学形式是：

        h = (I - U_out_bar U_out_bar^T) · B · A · (I - U_in_bar U_in_bar^T) · x

    这里：
    - x: 输入向量
    - A, B: 标准 LoRA 的两个低秩可训练矩阵
    - U_in_bar: 输入侧“高曲率方向”的正交基
    - U_out_bar: 输出侧“高曲率方向”的正交基
    - I - U U^T: 投影到“与高曲率方向正交”的子空间，也就是低曲率子空间

    直观上可以理解为：
    1. 先把输入 x 中“危险/敏感”的高曲率分量去掉
    2. 再做普通 LoRA
    3. 再把输出中“危险/敏感”的高曲率分量去掉

    因此，这个 LoRA 只能在更“平缓”的方向上更新模型。

    合并后的增量权重（delta weight）为：

        ΔW = (I - U_out_bar U_out_bar^T) @ B @ A @ (I - U_in_bar U_in_bar^T) * scaling

    其中 scaling 是 LoRA 的缩放系数。
    """

    @staticmethod
    def init(module: LoraLayer, adapter_name: str, **kwargs) -> None:
        """
        初始化当前 adapter 需要的“高曲率方向基”。

        参数：
        - module: 当前被注入 LoRA 的层，一般是一个线性层包装后的 LoraLayer
        - adapter_name: 当前 adapter 的名字，比如 "default"

        做的事情：
        1. 找到底层原始权重 base_weight
        2. 注册两个 buffer：
           - U_in_bar_{adapter_name}: 输入侧高曲率方向基
           - U_out_bar_{adapter_name}: 输出侧高曲率方向基

        为什么用 register_buffer？
        - 因为这些张量不是可训练参数，不需要 optimizer 更新
        - 但它们需要跟着模型一起保存/加载、搬到 GPU、切 dtype/device
        """

        # 有些 LoraLayer 会把原始层包在 base_layer 里；
        # 如果有 base_layer，就取 base_layer.weight
        # 否则直接取 module.weight
        base_weight = (
            module.base_layer.weight if hasattr(module, "base_layer") else module.weight
        )

        # 注册输入侧高曲率方向基 U_in_bar
        # 形状：(in_features, 0)
        # 为什么第二维先设成 0？
        # 因为初始化时还没有真正的特征向量，先放一个“空基”
        # 以后外部代码可以再把它替换成真正的特征向量矩阵
        module.register_buffer(
            f"U_in_bar_{adapter_name}",
            base_weight.new_zeros((module.in_features, 0), dtype=base_weight.dtype),
        )

        # 注册输出侧高曲率方向基 U_out_bar
        # 形状：(out_features, 0)
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
        """
        前向传播时，计算 CurvatureLora 的输出并加到原始 result 上。

        参数：
        - module: 当前 LoraLayer
        - active_adapter: 当前激活的 adapter 名称
        - x: 输入张量
             常见形状：
             * 对线性层而言可能是 (batch, seq_len, in_features)
             * 或者 (batch, in_features)
        - result: 原始基座模型的输出，即 base_layer(x)

        返回：
        - result + CurvatureLoRA增量

        数学过程：
        1. x_proj = (I - U_in_bar U_in_bar^T) x
        2. h = B(A(dropout(x_proj)))
        3. h_proj = (I - U_out_bar U_out_bar^T) h
        4. output = result + scaling * h_proj
        """

        # 取出当前 adapter 对应的输入/输出高曲率方向基
        U_in_bar = getattr(module, f"U_in_bar_{active_adapter}")
        U_out_bar = getattr(module, f"U_out_bar_{active_adapter}")

        # 取出当前 adapter 的 LoRA A/B 模块
        # 一般：
        # - lora_A: 把 in_features -> rank
        # - lora_B: 把 rank -> out_features
        lora_A = module.lora_A[active_adapter]
        lora_B = module.lora_B[active_adapter]

        # 取出当前 adapter 的 dropout 和 scaling
        dropout = module.lora_dropout[active_adapter]
        scaling = module.scaling[active_adapter]

        # ------------------------------
        # Step 1: 类型对齐
        # ------------------------------
        # 把 U_in_bar / U_out_bar 转成和输入 x 一样的 dtype
        # 例如 x 是 float16，那么投影基也转成 float16
        # 这样可以避免 matmul 时出现 dtype 不匹配
        U_in_bar = U_in_bar.to(dtype=x.dtype)
        U_out_bar = U_out_bar.to(dtype=x.dtype)

        # ------------------------------
        # Step 2: 输入侧投影
        # ------------------------------
        # 数学形式：
        #   x_proj = x - (x @ U_in_bar) @ U_in_bar^T
        #
        # 解释：
        # - x @ U_in_bar             -> 先把 x 投到高曲率基上，得到“在高曲率方向上的坐标”
        # - (x @ U_in_bar) @ U_in_bar^T
        #                           -> 再重构出 x 在高曲率子空间中的分量
        # - x - 这个分量             -> 去掉高曲率分量，只保留其正交补
        #
        # 如果 U_in_bar 的列向量是标准正交的，那么：
        #   U_in_bar U_in_bar^T
        # 就是投影到高曲率子空间的投影矩阵
        #
        # 因此：
        #   I - U_in_bar U_in_bar^T
        # 就是投影到“低曲率子空间”的投影矩阵
        #
        # 注意：
        # 如果 U_in_bar 形状是 (in_features, 0)，那么：
        # - x @ U_in_bar 的结果是空张量
        # - 后面整个减法等于 0
        # - 最终 x_proj = x
        # 也就是说，空基时不会出错，相当于“暂时不做输入投影”
        x_proj = x - (x @ U_in_bar) @ U_in_bar.T

        # ------------------------------
        # Step 3: 标准 LoRA 分支
        # ------------------------------
        # 标准 LoRA 计算过程：
        #   h = B(A(dropout(x_proj)))
        #
        # 其中：
        # - dropout(x_proj): LoRA 分支内部的 dropout
        # - A: 低秩降维，in_features -> r
        # - B: 低秩升维，r -> out_features
        #
        # 最终 h 的形状与原层输出一致（最后一维为 out_features）
        h = lora_B(lora_A(dropout(x_proj)))

        # ------------------------------
        # Step 4: 输出侧投影
        # ------------------------------
        # 数学形式：
        #   h_proj = h - (h @ U_out_bar) @ U_out_bar^T
        #
        # 逻辑和输入投影完全一样：
        # - 先提取 h 在输出高曲率方向上的分量
        # - 再减掉
        # - 只保留低曲率部分
        #
        # 因此 LoRA 最终只能在输出侧的低曲率子空间中产生更新
        h_proj = h - (h @ U_out_bar) @ U_out_bar.T

        # ------------------------------
        # Step 5: 加回主分支结果
        # ------------------------------
        # result 是原始层输出
        # h_proj * scaling 是 LoRA 增量
        #
        # scaling 通常等于 lora_alpha / r
        # 用于控制 LoRA 更新的幅度
        return result + h_proj * scaling

    @staticmethod
    def _compute_delta_weight(module: LoraLayer, active_adapter: str) -> torch.Tensor:
        """
        显式计算当前 CurvatureLoRA 对应的“等效权重增量” delta。

        数学表达式：

            ΔW = (I - U_out_bar U_out_bar^T) @ B @ A @ (I - U_in_bar U_in_bar^T) * scaling

        作用：
        - 在 merge 的时候，需要把 LoRA 分支真正折叠进原始权重 W 里
        - 这时不能只在 forward 时动态算，必须显式得到一个矩阵 delta

        返回：
        - delta，形状应与原始权重形状一致，通常是 (out_features, in_features)
        """

        # 取出输入/输出高曲率基
        U_in_bar = getattr(module, f"U_in_bar_{active_adapter}")
        U_out_bar = getattr(module, f"U_out_bar_{active_adapter}")

        # 取出 LoRA A/B 的权重矩阵
        # 对于 nn.Linear:
        # - lora_A.weight 形状一般为 (r, in_features)
        # - lora_B.weight 形状一般为 (out_features, r)
        weight_A = module.lora_A[active_adapter].weight
        weight_B = module.lora_B[active_adapter].weight

        # 缩放系数
        scaling = module.scaling[active_adapter]

        # 记录 device 和 dtype，后面可能需要临时转精度
        device = weight_B.device
        dtype = weight_B.dtype

        # ------------------------------
        # Step 1: CPU 低精度保护
        # ------------------------------
        # 在 CPU 上做 float16 / bfloat16 的矩阵乘法，常常不稳定或者不支持得很好
        # 因此如果当前在 CPU 且 dtype 是 fp16/bf16，就先临时转成 float32 计算
        cast_to_fp32 = device.type == "cpu" and (
            dtype == torch.float16 or dtype == torch.bfloat16
        )

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()
            U_in_bar = U_in_bar.float()
            U_out_bar = U_out_bar.float()
        else:
            # 如果不需要转 fp32，就只保证 U_in_bar / U_out_bar 的 dtype 与权重一致
            U_in_bar = U_in_bar.to(dtype=dtype)
            U_out_bar = U_out_bar.to(dtype=dtype)

        # ------------------------------
        # Step 2: 先算标准 LoRA 的增量 BA
        # ------------------------------
        # weight_B @ weight_A
        # 形状：
        #   (out_features, r) @ (r, in_features)
        # = (out_features, in_features)
        #
        # 这就是普通 LoRA 合并时的 delta weight（还没乘 scaling）
        BA = weight_B @ weight_A

        # ------------------------------
        # Step 3: 左侧投影（输出侧投影）
        # ------------------------------
        # 数学形式：
        #   P_out_BA = (I - U_out_bar U_out_bar^T) @ BA
        #
        # 展开后：
        #   BA - U_out_bar @ (U_out_bar^T @ BA)
        #
        # 形状检查：
        # - U_out_bar:               (out_features, k_out)
        # - U_out_bar.T @ BA:        (k_out, in_features)
        # - U_out_bar @ 上式:         (out_features, in_features)
        #
        # 这一步表示：从 BA 的“行空间/输出空间”里，去掉高曲率方向分量
        P_out_BA = BA - U_out_bar @ (U_out_bar.T @ BA)

        # ------------------------------
        # Step 4: 右侧投影（输入侧投影）
        # ------------------------------
        # 数学形式：
        #   delta = P_out_BA @ (I - U_in_bar U_in_bar^T)
        #
        # 展开后：
        #   delta = P_out_BA - (P_out_BA @ U_in_bar) @ U_in_bar^T
        #
        # 形状检查：
        # - P_out_BA @ U_in_bar:         (out_features, k_in)
        # - 上式 @ U_in_bar.T:           (out_features, in_features)
        #
        # 这一步表示：从 BA 的“列空间/输入空间”里，去掉高曲率方向分量
        delta = P_out_BA - (P_out_BA @ U_in_bar) @ U_in_bar.T

        # ------------------------------
        # Step 5: 乘上 LoRA 缩放系数
        # ------------------------------
        delta = delta * scaling

        # 如果之前为了 CPU 稳定性转成了 fp32，这里再转回原 dtype
        if cast_to_fp32:
            delta = delta.to(dtype=dtype)

        return delta

    @staticmethod
    def merge_safe(
        module: LoraLayer, active_adapter: str, orig_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        安全合并：返回一个“新的合并后权重”，但不原地修改 orig_weight。

        参数：
        - module: 当前 LoraLayer
        - active_adapter: 当前 adapter 名称
        - orig_weight: 原始权重 W

        返回：
        - W + ΔW

        为什么叫 safe？
        - 因为它不改动输入的 orig_weight 本体
        - 更安全，适合函数式调用或调试
        """
        delta = CurvatureLora._compute_delta_weight(module, active_adapter)

        # 保证 delta 的 dtype 与原权重一致，再相加
        return orig_weight + delta.to(orig_weight.dtype)

    @staticmethod
    def merge_unsafe(
        module: LoraLayer, active_adapter: str, orig_weight: torch.Tensor
    ) -> None:
        """
        非安全合并：直接原地修改 orig_weight.data。

        也就是：
            orig_weight <- orig_weight + ΔW

        为什么叫 unsafe？
        - 因为它是原地改动
        - 如果你之后还想保留未 merge 的权重状态，就不方便了
        - 直接操作 .data 也绕过了 autograd，一般只适合推理/部署场景
        """
        delta = CurvatureLora._compute_delta_weight(module, active_adapter)

        # 原地加法，直接改写原权重
        orig_weight.data += delta.to(orig_weight.dtype)

    @staticmethod
    def unmerge(
        module: LoraLayer, active_adapter: str, orig_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        取消合并：返回去掉 LoRA 增量后的权重。

        数学上：
            W = (W + ΔW) - ΔW

        返回：
        - orig_weight - delta

        注意：
        - 这里假设 orig_weight 当前是“已合并的权重”
        - 若传入的是未合并权重，再减一次就会错
        """
        delta = CurvatureLora._compute_delta_weight(module, active_adapter)
        return orig_weight - delta.to(orig_weight.dtype)