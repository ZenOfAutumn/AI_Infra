"""
基于 Torch FX 的模型剪枝与结构分析示例

演示以下功能：
1. 结构分析  - 遍历 FX 图，统计各层参数量和 FLOPs
2. 层类型分布 - 按 op 类型统计节点数量
3. 结构化剪枝 - 通过 FX Pass 删除指定名称的 Linear 层（零侵入）
4. 对比剪枝前后的模型参数量与推理速度
"""

import copy
import time

import torch
import torch.fx as fx
import torch.nn as nn


# ============================================================
# 模型定义：一个简单的多层 MLP，用于演示剪枝
# ============================================================

class DeepMLP(nn.Module):
    """四层 MLP，包含一个可被剪掉的"瓶颈层"（bottleneck）。"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.relu1 = nn.ReLU()
        # bottleneck：维度缩小后立刻扩大，实际贡献较小，是剪枝目标
        self.bottleneck = nn.Linear(256, 32)
        self.relu_bn = nn.ReLU()
        self.expand = nn.Linear(32, 256)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(256, 64)
        self.relu3 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.fc1(x))
        x = self.relu_bn(self.bottleneck(x))
        x = self.relu2(self.expand(x))
        x = self.relu3(self.fc2(x))
        return x


# ============================================================
# 功能 1：结构分析 —— 统计参数量与 FLOPs
# ============================================================

def analyze_graph(gm: fx.GraphModule, input_shape: tuple) -> dict:
    """
    遍历 FX 图，逐节点统计：
    - 每个 call_module 节点的参数量
    - 每个 Linear 层的 FLOPs（乘加运算次数，MAC = weight_out × weight_in）
    - 全图节点分布（按 op 类型）

    返回一个包含分析结果的字典。
    """
    named_modules = dict(gm.named_modules())
    total_params = 0
    total_flops = 0

    layer_stats = []   # 每层的详细信息
    op_distribution = {}  # op 类型 → 节点数量

    # 用符号形状追踪中间张量维度（简化版：只支持 Linear）
    # 真实输入形状为 (batch, *input_shape)
    batch_size = 1

    for node in gm.graph.nodes:
        # 统计 op 分布
        op_distribution[node.op] = op_distribution.get(node.op, 0) + 1

        if node.op != 'call_module':
            continue

        module = named_modules.get(node.target)
        if module is None:
            continue

        # 参数量统计
        params = sum(p.numel() for p in module.parameters())
        total_params += params

        # FLOPs 统计（仅支持 Linear）
        flops = 0
        if isinstance(module, nn.Linear):
            # 一次 Linear：output_features × input_features 次 MAC
            # 一次 MAC = 1 次乘法 + 1 次加法，通常计为 2 个 FLOP
            flops = 2 * module.in_features * module.out_features * batch_size
            total_flops += flops

        layer_stats.append({
            'node':   node.name,
            'target': node.target,
            'type':   type(module).__name__,
            'params': params,
            'flops':  flops,
        })

    return {
        'layer_stats':    layer_stats,
        'total_params':   total_params,
        'total_flops':    total_flops,
        'op_distribution': op_distribution,
    }


def print_analysis(stats: dict, title: str = "模型结构分析"):
    """格式化打印分析结果。"""
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")
    print(f"  {'节点名':<22} {'模块类型':<18} {'参数量':>10} {'FLOPs':>12}")
    print(f"  {'-'*62}")
    for s in stats['layer_stats']:
        flops_str = f"{s['flops']:,}" if s['flops'] > 0 else '-'
        print(f"  {s['node']:<22} {s['type']:<18} {s['params']:>10,} {flops_str:>12}")
    print(f"  {'-'*62}")
    print(f"  {'合计':<22} {'':<18} {stats['total_params']:>10,} {stats['total_flops']:>12,}")
    print(f"\n  节点类型分布:")
    for op, count in sorted(stats['op_distribution'].items()):
        print(f"    {op:<20} : {count} 个节点")


# ============================================================
# 功能 2：结构化剪枝 Pass —— 删除指定的 call_module 层链
# ============================================================

def pass_prune_layers(
    graph_module: fx.GraphModule,
    prune_targets: set,
) -> fx.GraphModule:
    """
    FX 剪枝 Pass：删除 prune_targets 中指定的模块节点及其直接前驱激活函数。

    核心策略（跳连替换，Skip-Connection Bypass）：
      将被删节点的所有「下游使用者」改为直接使用该节点的「上游输入」，
      相当于在图中建立了一条跳连，绕过被删层。

    Args:
        graph_module: 待剪枝的 FX GraphModule
        prune_targets: 需要删除的模块名称集合，如 {'bottleneck', 'relu_bn', 'expand', 'relu2'}
    """
    gm = copy.deepcopy(graph_module)
    named_modules = dict(gm.named_modules())

    # 按拓扑序（图的反向）收集待删节点，确保先删叶节点
    nodes_to_remove = []
    for node in gm.graph.nodes:
        if node.op == 'call_module' and node.target in prune_targets:
            nodes_to_remove.append(node)

    for node in nodes_to_remove:
        # 找到该节点的第一个输入（上游节点）
        if not node.args:
            continue
        upstream = node.args[0]

        # 将所有使用该节点输出的地方替换为直接使用上游输入
        node.replace_all_uses_with(upstream)

    # 删除节点（需从后往前，避免先删前驱导致后继节点引用悬空）
    for node in reversed(nodes_to_remove):
        if len(node.users) == 0:  # 确认已无使用者，安全删除
            gm.graph.erase_node(node)

    # 也从 module 字典中移除对应的子模块（不影响图执行，但保持结构干净）
    for target in prune_targets:
        # target 可能是 "bottleneck" 或 "sub.bottleneck" 等层级路径
        parts = target.split('.')
        parent = gm
        for part in parts[:-1]:
            parent = getattr(parent, part, parent)
        if hasattr(parent, parts[-1]):
            delattr(parent, parts[-1])

    gm.recompile()
    return gm


# ============================================================
# 功能 3：性能对比
# ============================================================

def benchmark(model: nn.Module, x: torch.Tensor, n_warmup: int = 50, n_run: int = 500) -> float:
    """测量模型推理的平均耗时（毫秒）。"""
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        start = time.perf_counter()
        for _ in range(n_run):
            model(x)
        end = time.perf_counter()
    return (end - start) / n_run * 1000


# ============================================================
# 主流程
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 256
    x = torch.randn(batch_size, 128)

    model = DeepMLP()
    model.eval()

    # --- Step 1: 捕获 FX 图 ---
    print("\n>>> Step 1: 使用 Torch FX 捕获计算图")
    traced = fx.symbolic_trace(model)
    print(f"  已捕获 {len(list(traced.graph.nodes))} 个节点")

    # --- Step 2: 结构分析 ---
    print("\n>>> Step 2: 结构分析（参数量 & FLOPs）")
    stats_before = analyze_graph(traced, input_shape=(128,))
    print_analysis(stats_before, "剪枝前 —— DeepMLP 结构分析")

    # --- Step 3: 结构化剪枝（删除 bottleneck → relu_bn → expand → relu2 链路）---
    print("\n>>> Step 3: 执行结构化剪枝")
    print("  剪枝目标: bottleneck → relu_bn → expand → relu2（瓶颈层链路）")

    # 注意：prune_targets 中的顺序无关，Pass 内部会按拓扑序处理
    prune_targets = {'bottleneck', 'relu_bn', 'expand', 'relu2'}
    traced_pruned = pass_prune_layers(traced, prune_targets)
    print(f"  剪枝后节点数: {len(list(traced_pruned.graph.nodes))}")

    # --- Step 4: 剪枝后结构分析 ---
    print("\n>>> Step 4: 剪枝后结构分析")
    stats_after = analyze_graph(traced_pruned, input_shape=(128,))
    print_analysis(stats_after, "剪枝后 —— DeepMLP 结构分析")

    # 参数量对比
    p_before = stats_before['total_params']
    p_after  = stats_after['total_params']
    f_before = stats_before['total_flops']
    f_after  = stats_after['total_flops']
    print(f"\n  参数量: {p_before:,} → {p_after:,}  (减少 {(1 - p_after/p_before)*100:.1f}%)")
    print(f"  FLOPs : {f_before:,} → {f_after:,}  (减少 {(1 - f_after/f_before)*100:.1f}%)")

    # --- Step 5: 数值验证（剪枝改变了网络结构，输出形状须一致）---
    print("\n>>> Step 5: 验证剪枝后输出形状")
    with torch.no_grad():
        out_before = model(x)
        out_after  = traced_pruned(x)
    print(f"  剪枝前输出形状: {tuple(out_before.shape)}")
    print(f"  剪枝后输出形状: {tuple(out_after.shape)}")
    shapes_match = out_before.shape == out_after.shape
    print(f"  输出形状一致: {'✅' if shapes_match else '❌'}")
    # 注意：剪枝改变了网络权重路径，输出数值会不同（需重新微调/蒸馏恢复精度）

    # --- Step 6: 推理性能对比 ---
    print("\n>>> Step 6: 推理性能对比")
    t_before = benchmark(model,         x)
    t_fx_orig = benchmark(traced,       x)
    t_pruned  = benchmark(traced_pruned, x)

    print(f"\n  {'模型':<30} {'平均耗时 (ms)'}")
    print(f"  {'-'*48}")
    print(f"  {'原始 DeepMLP':<30} {t_before:.4f} ms")
    print(f"  {'FX Traced (未剪枝)':<30} {t_fx_orig:.4f} ms")
    print(f"  {'FX Traced (剪枝后)':<30} {t_pruned:.4f} ms")
    print(f"\n  相比原始模型，剪枝后加速比: {t_before / t_pruned:.2f}x")

