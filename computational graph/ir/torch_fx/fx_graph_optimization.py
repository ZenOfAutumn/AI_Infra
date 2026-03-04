"""
Torch FX Graph 优化演示

本示例实现了一个两层 MLP 神经网络，使用 Torch FX 捕获其计算图，
并实现以下两个图优化 Pass：

1. Dead Code Elimination (DCE) - 死代码消除
2. Operator Fusion (算子融合) - 将 Linear + ReLU 融合为单一算子，
   减少显存读写次数，提升执行效率。

然后对比优化前后的计算图结构和推理性能。
"""

import copy
import time

import matplotlib

matplotlib.use('Agg')  # 非交互式后端，将图形保存为文件
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager

# 配置中文字体：优先使用 macOS 系统内置字体，fallback 到 DejaVu Sans
_CN_FONT_CANDIDATES = ['PingFang SC', 'Heiti TC', 'STHeiti', 'Arial Unicode MS']
_cn_font = None
for _candidate in _CN_FONT_CANDIDATES:
    try:
        _found = font_manager.findfont(_candidate, fallback_to_default=False)
        if _found:
            _cn_font = _candidate
            break
    except (ValueError, Exception):
        continue
if _cn_font:
    plt.rcParams['font.sans-serif'] = [_cn_font, 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示
import torch
import torch.fx as fx
import torch.nn as nn


# ============================================================
# 第一步：定义前端高层语言 (Python/PyTorch) 的神经网络模型
# ============================================================

class MLP(nn.Module):
    """一个包含冗余操作的两层 MLP，用于展示 FX 优化效果。"""

    def __init__(self, input_dim=128, hidden_dim=256, output_dim=64):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu1    = nn.ReLU()
        self.linear2  = nn.Linear(hidden_dim, output_dim)
        self.relu2    = nn.ReLU()

        # 刻意加入一个"无用"的 dropout（eval 模式下恒等变换），
        # 用于演示死代码消除（Dead Code Elimination）的效果。
        self.dropout  = nn.Dropout(p=0.0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        # 在 eval 模式下，dropout(p=0.0) 是一个无操作
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu2(x)
        return x


# ============================================================
# 第二步：使用 Torch FX 捕获计算图
# ============================================================

def capture_fx_graph(model: nn.Module, sample_input: torch.Tensor) -> fx.GraphModule:
    """使用 torch.torch_fx.symbolic_trace 将 nn.Module 转换为 FX GraphModule。"""
    model.eval()
    # symbolic_trace 会遍历 forward() 的所有 Python 代码，
    # 用符号值（Proxy）追踪每一步操作，生成一个完整的计算图（Graph）。
    traced: fx.GraphModule = fx.symbolic_trace(model)
    return traced


# ============================================================
# 第三步：实现两个图优化 Pass
# ============================================================

def pass_dead_code_elimination(graph_module: fx.GraphModule) -> fx.GraphModule:
    """
    优化 Pass 1: 死代码消除 (Dead Code Elimination)

    检查图中每一个节点：如果它的输出没有被任何其他节点使用，
    且它不是最终输出（'output' 节点），则将其删除。

    本示例中，eval 模式下 Dropout(p=0.0) 虽然在 Python 层面
    不改变值，但 FX 仍然会为其生成一个图节点。
    通过 torch.torch_fx.passes.dead_code_elimination 可消除它。
    """
    gm = copy.deepcopy(graph_module)
    # 注意：torch.torch_fx 内置了 DCE，我们直接调用
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


def pass_fuse_linear_relu(graph_module: fx.GraphModule) -> fx.GraphModule:
    """
    优化 Pass 2: Linear + ReLU 算子融合

    遍历图中所有节点，找到 "Linear -> ReLU" 的组合模式，
    将其替换为一个自定义的 FusedLinearReLU 模块。

    融合的意义：
    - 未融合：Linear 的输出写入显存 -> ReLU 再从显存读取 -> ReLU 的输出写入显存
    - 已融合：Linear 计算完成后，ReLU 直接在寄存器中处理，最终写一次显存
    """

    class FusedLinearReLU(nn.Module):
        """融合后的 Linear + ReLU 单一模块。"""
        def __init__(self, linear: nn.Linear):
            super().__init__()
            self.linear = linear

        def forward(self, x):
            return torch.relu(self.linear(x))

    gm = copy.deepcopy(graph_module)

    # 构建节点名到模块的映射，方便查找
    node_to_module = dict(gm.named_modules())

    fuse_count = 0
    nodes_to_erase = []

    for node in list(gm.graph.nodes):
        # 寻找 ReLU 节点（call_module 类型）
        if node.op != 'call_module':
            continue

        module = node_to_module.get(node.target)
        if not isinstance(module, nn.ReLU):
            continue

        # 找到 ReLU 的输入节点，检查是否是 Linear
        if len(node.args) != 1:
            continue

        prev_node = node.args[0]
        if prev_node.op != 'call_module':
            continue

        prev_module = node_to_module.get(prev_node.target)
        if not isinstance(prev_module, nn.Linear):
            continue

        # 找到 Linear -> ReLU 模式，开始融合
        fused_name = f"fused_linear_relu_{fuse_count}"
        fused_module = FusedLinearReLU(prev_module)
        gm.add_module(fused_name, fused_module)

        # 在图中，用融合节点替换 Linear 节点（ReLU 节点的前驱）
        with gm.graph.inserting_after(prev_node):
            fused_node = gm.graph.call_module(
                fused_name,
                args=(prev_node.args[0],),  # Linear 的原始输入
            )

        # 将所有使用 ReLU 输出的节点，改为使用融合节点的输出
        node.replace_all_uses_with(fused_node)

        # 标记 Linear 和 ReLU 节点待删除
        nodes_to_erase.append(node)
        nodes_to_erase.append(prev_node)

        # 更新模块映射
        node_to_module[fused_name] = fused_module
        fuse_count += 1

    # 删除被替换的旧节点
    for n in nodes_to_erase:
        gm.graph.erase_node(n)

    gm.recompile()
    return gm


# ============================================================
# 第四步：对比优化前后的计算图结构和推理性能
# ============================================================

def benchmark(model: nn.Module, x: torch.Tensor, n_warmup=50, n_run=500) -> float:
    """测量模型推理的平均耗时（毫秒）。"""
    model.eval()
    with torch.no_grad():
        # 预热
        for _ in range(n_warmup):
            model(x)
        # 正式计时
        start = time.perf_counter()
        for _ in range(n_run):
            model(x)
        end = time.perf_counter()
    return (end - start) / n_run * 1000  # ms


def print_graph(gm: fx.GraphModule, title: str):
    """打印 FX 计算图的所有节点。"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"{'节点名称':<35} {'操作类型':<16} {'目标'}")
    print(f"{'-'*80}")
    for node in gm.graph.nodes:
        print(f"{node.name:<35} {node.op:<16} {node.target}")
    print(f"{'-'*80}")
    print(f"共 {len(list(gm.graph.nodes))} 个节点")


# ============================================================
# 可视化辅助函数（须在 __main__ 块之前定义）
# ============================================================

def _build_layout(gm: fx.GraphModule):
    """将 FX 图的节点转为 (名称, 操作类型) 列表，按执行顺序排列。"""
    nodes = []
    for node in gm.graph.nodes:
        nodes.append((node.name, node.op))
    return nodes


# 节点类型 → 颜色映射
_OP_COLORS = {
    'placeholder': '#4CAF50',   # 输入：绿色
    'call_module': '#2196F3',   # 模块调用：蓝色
    'call_function': '#FF9800', # 函数调用：橙色
    'call_method': '#9C27B0',   # 方法调用：紫色
    'output': '#F44336',        # 输出：红色
    'get_attr': '#795548',      # 属性获取：棕色
}


def _is_fused(name: str) -> bool:
    return name.startswith('fused_')


def _visualize_graphs(
    original: fx.GraphModule,
    fused: fx.GraphModule,
    t_orig: float,
    t_fused: float,
):
    """并排绘制优化前后的计算图，保存为 PNG。"""
    orig_nodes  = _build_layout(original)
    fused_nodes = _build_layout(fused)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(len(orig_nodes), len(fused_nodes)) * 1.4 + 3))
    fig.patch.set_facecolor('#1E1E2E')

    def draw_graph(ax, nodes, title, t_ms):
        ax.set_facecolor('#1E1E2E')
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(nodes) - 0.5)
        ax.axis('off')

        n = len(nodes)
        node_y = {name: (n - 1 - i) for i, (name, _) in enumerate(nodes)}

        # 绘制有向箭头（竖向连接）
        for i in range(len(nodes) - 1):
            y_start = node_y[nodes[i][0]]
            y_end   = node_y[nodes[i + 1][0]]
            ax.annotate(
                '', xy=(0.5, y_end + 0.38), xytext=(0.5, y_start - 0.38),
                arrowprops=dict(arrowstyle='->', color='#AAAAAA', lw=1.8),
            )

        # 绘制节点矩形
        for name, op in nodes:
            y = node_y[name]
            color = '#FF5722' if _is_fused(name) else _OP_COLORS.get(op, '#607D8B')
            fancy = mpatches.FancyBboxPatch(
                (0.08, y - 0.35), 0.84, 0.7,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor='white', linewidth=1.2,
                zorder=3,
            )
            ax.add_patch(fancy)
            # 节点名称（大字）
            ax.text(0.5, y + 0.07, name, ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white', zorder=4)
            # 操作类型（小字）
            ax.text(0.5, y - 0.16, f'[{op}]', ha='center', va='center',
                    fontsize=7.5, color='#CCCCCC', zorder=4)

        # 标题
        ax.set_title(
            f'{title}\n节点数: {len(nodes)}   推理耗时: {t_ms:.4f} ms',
            color='white', fontsize=12, fontweight='bold', pad=12,
        )

    draw_graph(axes[0], orig_nodes,  '优化前  (Original Graph)', t_orig)
    draw_graph(axes[1], fused_nodes, '优化后  (Fused Graph)',    t_fused)

    # 图例
    legend_items = [
        mpatches.Patch(color=_OP_COLORS['placeholder'],  label='Input (placeholder)'),
        mpatches.Patch(color=_OP_COLORS['call_module'],  label='Module call'),
        mpatches.Patch(color='#FF5722',                  label='Fused operator [*]'),
        mpatches.Patch(color=_OP_COLORS['output'],       label='Output'),
    ]
    fig.legend(handles=legend_items, loc='lower center', ncol=4,
               facecolor='#2D2D3F', edgecolor='white',
               labelcolor='white', fontsize=9, bbox_to_anchor=(0.5, 0.01))

    plt.suptitle('Torch FX Graph Optimization — Before vs After',
                 color='white', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.07, 1, 0.96])

    save_path = 'computational graph/ir/fx_graph_compare.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    # --- 初始化 ---
    torch.manual_seed(42)
    batch_size = 512
    input_dim, hidden_dim, output_dim = 128, 256, 64
    x = torch.randn(batch_size, input_dim)

    model = MLP(input_dim, hidden_dim, output_dim)
    model.eval()

    # --- Step 1: FX 捕获原始计算图 ---
    print("\n>>> Step 1: 使用 Torch FX 捕获原始计算图")
    traced_original = capture_fx_graph(model, x)
    print_graph(traced_original, "原始计算图 (Original Graph)")

    # --- Step 2: 应用死代码消除 ---
    print("\n>>> Step 2: 应用 Pass 1 - 死代码消除 (Dead Code Elimination)")
    traced_dce = pass_dead_code_elimination(traced_original)
    print_graph(traced_dce, "DCE 后的计算图")

    # --- Step 3: 应用算子融合 ---
    print("\n>>> Step 3: 应用 Pass 2 - Linear + ReLU 算子融合")
    traced_fused = pass_fuse_linear_relu(traced_dce)
    print_graph(traced_fused, "融合后的计算图 (Fused Graph)")

    # --- Step 4: 验证数值正确性 ---
    print("\n>>> Step 4: 验证数值正确性")
    with torch.no_grad():
        out_original = model(x)
        out_fused    = traced_fused(x)
    is_close = torch.allclose(out_original, out_fused, atol=1e-5)
    print(f"  融合前后输出是否一致 (allclose): {is_close} ✅" if is_close else f"  ❌ 输出不一致！最大误差: {(out_original - out_fused).abs().max()}")

    # --- Step 5: 性能对比 ---
    print("\n>>> Step 5: 性能对比")
    t_original = benchmark(model,         x)
    t_traced   = benchmark(traced_original, x)
    t_fused    = benchmark(traced_fused,  x)

    print(f"\n  {'模型':<30} {'平均耗时 (ms)'}")
    print(f"  {'-'*45}")
    print(f"  {'原始 nn.Module':<30} {t_original:.4f} ms")
    print(f"  {'FX Traced (未优化)':<30} {t_traced:.4f} ms")
    print(f"  {'FX Traced (融合优化后)':<30} {t_fused:.4f} ms")
    print(f"\n  相比原始模型，融合优化加速比: {t_original / t_fused:.2f}x")
    print()

    # --- Step 6: 可视化计算图 ---
    print("\n>>> Step 6: 生成计算图可视化对比图")
    _visualize_graphs(traced_original, traced_fused, t_original, t_fused)
    print("  图形已保存至: computational graph/ir/fx_graph_compare.png")

