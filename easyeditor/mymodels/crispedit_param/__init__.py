from .utils import *
from .CrispEditParam_hparams import CrispEditParamHyperParams

# =============================================================================
# 数据流（Data Flow）
# =============================================================================
#
#  yaml 配置文件
#       │
#       ▼
#  CrispEditParamHyperParams.from_hparams()
#       │  layers / rewrite_module_tmp → 确定目标层
#       │  energy_threshold / mom2_* → 控制 KFAC 统计与投影
#       │  lr / weight_decay / num_steps → 优化超参数
#       │
#  ─── 阶段 1：冻结模型，只解冻目标层 ──────────────────────────────────────
#
#  get_weights(model, hparams, bias=False)
#       │  model.named_parameters() 筛选出 rewrite_module_tmp 对应的权重 W
#       │  返回 {layer_name: param_tensor}（param 本身，非拷贝）
#       │
#  外部调用方对 model 其余参数调用 .requires_grad_(False)，只保留 W 可训练
#
#  ─── 阶段 2：计算 KFAC 协方差缓存 ───────────────────────────────────────
#
#  [路径 A] 预训练数据
#  calculate_cov_cache_with_old_data(model, tok, hparams)
#       │  layer_stats_kfac_one_pass() → stats_dict
#       ▼
#  layer_to_cov_cache: {layer_name → {A, B, num_samples}}
#
#  [路径 B] 编辑请求样本（每次编辑调用）
#  calculate_cov_cache_with_request(txt, tgt, model, tok, hparams)
#       ▼
#  layer_to_cov_cache: {layer_name → {A, B, num_samples}}
#
#  ─── 阶段 3：构建 ProjectedAdam（参数变化投影约束）──────────────────────
#
#  build_optimizer_with_cov_caches(model, hparams, [cov_cache_list])
#       │
#       ├─ combine_layer_to_cov_caches()   多组缓存按样本数加权平均
#       │
#       ├─ calculate_projection_caches_from_cov_caches()
#       │    对每层：
#       │    │  calculate_projection_cache_with_kfac(A, B, energy_threshold)
#       │    │    Sa, Ua = eigh(A)    Sb, Ub = eigh(B)
#       │    │    M_energy = outer(Sa, Sb)
#       │    │    M = M_energy < null_threshold   ← 低曲率掩码（True=安全方向）
#       │    │    返回 {Ua, Ub, M}
#       │    └─ 以 weight_param 对象为键 → weight_to_projection_cache
#       │
#       └─ ProjectedAdam(weights, projection_cache_map)
#            step() 时对每个 W 的梯度执行：
#              grad_proj = Ub @ ( (Ub.T @ grad @ Ua) * M.T ) @ Ua.T
#            即：把梯度（ΔW 方向）投影到低曲率子空间
#            高曲率方向（旧知识敏感区域）的梯度分量被置零
#
#  ─── 阶段 4：训练循环（外部驱动）────────────────────────────────────────
#
#  for step in range(num_steps):
#    loss = model(input_ids, labels) → backward()
#    opt.step()   # ProjectedAdam 在此处做曲率投影
#    opt.zero_grad()
#
#  ─── 阶段 5：权重变化检测（可选，连续编辑）──────────────────────────────
#
#  recalculate_cov_cache_if_weights_changed(model, tok, hparams,
#                                            current_weights_cpu, layer_to_cov_cache)
#       │  is_weights_changed() 计算相对范数变化 > recalculate_weight_threshold?
#       │    是 → calculate_cov_cache_with_old_data(force_recompute=True)
#       │         build_optimizer_with_cov_caches(..., opt=opt)   # reset 投影缓存
#       ▼
#  (new_weights_cpu, new_cov_cache, recalculated: bool)
#
#  输出：直接修改后的模型原始权重 W（无 LoRA 附加结构）
#
# =============================================================================
