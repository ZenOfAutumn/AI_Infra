"""
Torch FX 进阶优化 Pass 示例

演示以下四种常见图优化 Pass：
1. 常量折叠（Constant Folding）        — 预计算全由常量构成的子图
2. 代数简化与算子替换（Algebraic Simplification） — 用等价但更高效的算子替换原始算子
3. 公共子表达式消除（Common Subexpression Elimination, CSE） — 消除重复计算
4. 静态内存规划（Static Memory Planning） — 分析张量生命周期，最小化峰值显存占用
"""

import copy
from typing import Dict, Any

import torch
import torch.fx as fx
import torch.nn as nn


# ============================================================
# 辅助：打印图节点摘要
# ============================================================

def print_graph(gm: fx.GraphModule, title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  {'节点名':<28} {'op':<16} {'target'}")
    print(f"  {'-'*58}")
    for node in gm.graph.nodes:
        print(f"  {node.name:<28} {node.op:<16} {node.target}")
    print(f"  {'-'*58}")
    print(f"  共 {len(list(gm.graph.nodes))} 个节点")


# ============================================================
# Pass 1：常量折叠（Constant Folding）
# ============================================================
#
# 原理：如果一个节点的所有输入都是常量（get_attr / prim::Constant），
# 那么它的输出也是固定的——可以在编译期直接执行，将结果作为新的常量
# 写入模型属性，替换掉原始计算节点，从而避免运行时的重复计算。
#
# 典型场景：
#   - 模型中硬编码的 scale / shift 参数（如 LayerNorm 的 eps）
#   - 预处理中固定的归一化系数 (mean, std)
#   - 推理时固定不变的位置编码（PositionalEncoding）
# ============================================================

class NormModel(nn.Module):
    """
    模拟一个含常量归一化的模型：
      out = (x - MEAN) / STD
    MEAN 和 STD 在模型初始化时固定，属于"常量"，
    常量折叠可将 sub/div 提前计算为单个 scale/shift。
    """
    def __init__(self):
        super().__init__()
        # 注册为 buffer，FX 追踪时会产生 get_attr 节点
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def pass_constant_folding(gm: fx.GraphModule) -> fx.GraphModule:
    """
    常量折叠 Pass。

    遍历图节点，找到"所有输入均来自 get_attr（常量属性）"的
    call_function / call_method 节点，在图外提前执行它，
    将结果注册为新的 buffer，并用 get_attr 节点替换原节点。
    """
    gm = copy.deepcopy(gm)

    # 收集所有 get_attr 节点对应的真实值
    const_values: Dict[str, Any] = {}
    for node in gm.graph.nodes:
        if node.op == 'get_attr':
            # 按属性路径取值
            val = gm
            for part in node.target.split('.'):
                val = getattr(val, part)
            const_values[node.name] = val

    # 找到可折叠节点
    nodes_to_fold = []
    for node in gm.graph.nodes:
        if node.op not in ('call_function', 'call_method'):
            continue
        # 检查所有位置参数是否都是常量
        all_const = all(
            (isinstance(a, fx.Node) and a.name in const_values) or not isinstance(a, fx.Node)
            for a in node.args
        )
        if not all_const:
            continue
        nodes_to_fold.append(node)

    folded_count = 0
    for node in nodes_to_fold:
        # 解析参数的真实值
        def resolve(a):
            if isinstance(a, fx.Node):
                return const_values.get(a.name)
            return a

        args_vals = tuple(resolve(a) for a in node.args)
        kwargs_vals = {k: resolve(v) for k, v in node.kwargs.items()}

        # 提前执行
        try:
            if node.op == 'call_function':
                result = node.target(*args_vals, **kwargs_vals)
            else:  # call_method
                obj = args_vals[0]
                result = getattr(obj, node.target)(*args_vals[1:], **kwargs_vals)
        except Exception:
            continue  # 无法折叠则跳过

        if not isinstance(result, torch.Tensor):
            continue

        # 将结果注册为新 buffer
        buf_name = f'_folded_{folded_count}'
        gm.register_buffer(buf_name, result)
        const_values[node.name] = result
        folded_count += 1

        # 用 get_attr 节点替换原计算节点
        with gm.graph.inserting_before(node):
            new_node = gm.graph.get_attr(buf_name)
        node.replace_all_uses_with(new_node)
        gm.graph.erase_node(node)

    gm.recompile()
    print(f"  [常量折叠] 共折叠 {folded_count} 个节点")
    return gm


# ============================================================
# Pass 2：代数简化与算子替换（Algebraic Simplification）
# ============================================================
#
# 原理：利用数学等价关系，将计算量更大或执行更慢的算子替换为等价的
# 更高效形式。常见规则：
#   - x * 1  →  x                  （乘 1 消除）
#   - x + 0  →  x                  （加 0 消除）
#   - x ** 2 →  x * x              （幂运算展开，部分后端更快）
#   - torch.div(x, c) → x * (1/c)  （除法变乘法，GPU 上乘法更快）
#   - relu(relu(x)) → relu(x)      （幂等性消除）
#
# 本示例演示：将 x / scalar 替换为 x * (1/scalar)，
# 以及消除 relu(relu(x)) 冗余双重激活。
# ============================================================

class AlgebraModel(nn.Module):
    """含代数冗余的模型：两次 relu + 一次除法。"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()   # relu(relu(x)) 冗余

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.relu1(x)
        x = self.relu2(x)        # 冗余 relu
        x = torch.div(x, 2.0)   # 除以常数 → 乘以 0.5 更快
        return x


def pass_algebraic_simplification(gm: fx.GraphModule) -> fx.GraphModule:
    """
    代数简化 Pass：
      1. relu(relu(x)) → relu(x)：检测相邻的两个 ReLU call_module，消除外层
      2. torch.div(x, scalar) → torch.mul(x, 1/scalar)：除法换乘法
    """
    gm = copy.deepcopy(gm)
    named_modules = dict(gm.named_modules())
    simplified = 0

    for node in list(gm.graph.nodes):
        # --- 规则 1：消除冗余 relu ---
        if (node.op == 'call_module'
                and isinstance(named_modules.get(node.target), nn.ReLU)
                and len(node.args) == 1):
            prev = node.args[0]
            if (prev.op == 'call_module'
                    and isinstance(named_modules.get(prev.target), nn.ReLU)):
                # node 是多余的外层 relu，直接跳过它
                node.replace_all_uses_with(prev)
                gm.graph.erase_node(node)
                simplified += 1
                print(f"  [代数简化] 消除冗余 relu: {node.name} → 直接使用 {prev.name}")
                continue

        # --- 规则 2：torch.div(x, scalar) → torch.mul(x, 1/scalar) ---
        if (node.op == 'call_function'
                and node.target is torch.div
                and len(node.args) == 2
                and isinstance(node.args[1], (int, float))):
            scalar = node.args[1]
            if scalar != 0:
                inv = 1.0 / scalar
                with gm.graph.inserting_before(node):
                    new_node = gm.graph.call_function(
                        torch.mul, args=(node.args[0], inv)
                    )
                node.replace_all_uses_with(new_node)
                gm.graph.erase_node(node)
                simplified += 1
                print(f"  [代数简化] div({scalar}) → mul({inv:.6f}): {node.name}")

    gm.recompile()
    print(f"  [代数简化] 共简化 {simplified} 处")
    return gm


# ============================================================
# Pass 3：公共子表达式消除（CSE, Common Subexpression Elimination）
# ============================================================
#
# 原理：如果图中存在两个完全相同的计算（相同算子 + 相同输入），
# 则只需计算一次，将后续所有引用指向第一次的结果，
# 消除冗余的重复计算，节省运算量和显存。
#
# 典型场景：
#   - 注意力机制中对同一 Q/K/V 做多次相同 reshape
#   - 多分支网络中各分支共享的前置特征变换
#   - 编码器对同一输入多次调用相同的归一化
# ============================================================

class CSEModel(nn.Module):
    """
    含公共子表达式的模型：
    对同一输入 x 做两次完全相同的 layer_norm，然后各自接一个 linear。
    layer_norm 只需算一次。
    """
    def __init__(self):
        super().__init__()
        self.norm   = nn.LayerNorm(64)
        self.linear_a = nn.Linear(64, 32)
        self.linear_b = nn.Linear(64, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 两次相同的 norm 调用 —— 就是公共子表达式
        normed_a = self.norm(x)
        normed_b = self.norm(x)
        return self.linear_a(normed_a) + self.linear_b(normed_b)


def pass_cse(gm: fx.GraphModule) -> fx.GraphModule:
    """
    公共子表达式消除 Pass。

    为每个节点计算"签名"（op + target + args_名称元组），
    若两个节点签名完全相同，则后者可以直接复用前者的输出。
    """
    gm = copy.deepcopy(gm)

    # 签名 → 第一次出现的节点
    seen: Dict[tuple, fx.Node] = {}
    cse_count = 0

    for node in list(gm.graph.nodes):
        if node.op in ('placeholder', 'output', 'get_attr'):
            continue  # 这些节点不参与 CSE

        # 构造签名：(op, target, args中每个元素的name或值, kwargs)
        def arg_key(a):
            if isinstance(a, fx.Node):
                return ('node', a.name)
            return ('const', a)

        sig = (
            node.op,
            node.target,
            tuple(arg_key(a) for a in node.args),
            tuple(sorted((k, arg_key(v)) for k, v in node.kwargs.items())),
        )

        if sig in seen:
            # 发现重复，替换为第一次的结果
            first = seen[sig]
            node.replace_all_uses_with(first)
            gm.graph.erase_node(node)
            cse_count += 1
            print(f"  [CSE] {node.name} 与 {first.name} 完全相同，已消除")
        else:
            seen[sig] = node

    gm.recompile()
    print(f"  [CSE] 共消除 {cse_count} 个冗余节点")
    return gm


# ============================================================
# Pass 4：静态内存规划（Static Memory Planning）
# ============================================================
#
# 原理：在图执行之前，分析每个中间张量的"生命周期"
# （从首次产生到最后一次被使用），找出生命周期不重叠的张量，
# 让它们共享同一块内存缓冲区（Buffer Reuse），从而降低峰值显存占用。
#
# 示意图：
#   时间步:   t0   t1   t2   t3   t4
#   tensor_A: [创建─────────────使用完]
#   tensor_B:           [创建─────────使用完]
#   → tensor_A 和 tensor_B 生命周期不重叠，可共享同一块内存。
#
# 注意：FX 本身不执行物理内存分配（那是后端算子库/运行时的工作），
# 此 Pass 执行的是"分析与标注"，将可共享的节点对输出到日志，
# 实际复用需结合 torch.compile 或自定义 Runtime 完成。
# ============================================================

class MemPlanModel(nn.Module):
    """
    用于内存规划分析的模型：线性激活序列，
    中间张量有清晰的生命周期边界。
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.relu(self.fc1(x))   # a: 256 维
        b = torch.relu(self.fc2(a))   # b: 128 维，a 之后不再使用
        c = self.fc3(b)               # c: 64 维，b 之后不再使用
        return c


def pass_static_memory_planning(gm: fx.GraphModule) -> dict:
    """
    静态内存规划 Pass（分析阶段）。

    计算每个中间张量的：
      - first_use: 首次被创建（产生）的时间步
      - last_use:  最后一次被读取（消费）的时间步

    找出生命周期不重叠的节点对，标注为"可共享内存"候选。

    返回分析报告字典，不修改图结构。
    """
    nodes = list(gm.graph.nodes)
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}

    # 计算每个节点的生命周期
    lifetimes: Dict[str, dict] = {}
    for i, node in enumerate(nodes):
        if node.op in ('placeholder', 'output'):
            continue
        # 首次产生 = 当前时间步
        first = i
        # 最后被使用 = 所有 user 中最大的时间步
        last = i
        for user in node.users:
            last = max(last, node_index.get(user, i))
        lifetimes[node.name] = {'first': first, 'last': last, 'node': node}

    # 找出不重叠的节点对（候选内存复用）
    reuse_candidates = []
    names = list(lifetimes.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = lifetimes[names[i]]
            b = lifetimes[names[j]]
            # 不重叠条件：a 结束后 b 才开始，或 b 结束后 a 才开始
            if a['last'] < b['first'] or b['last'] < a['first']:
                reuse_candidates.append((names[i], names[j],
                                         a['last'], b['first']))

    return {'lifetimes': lifetimes, 'reuse_candidates': reuse_candidates}


def print_memory_plan(report: dict):
    """格式化打印内存规划分析结果。"""
    lifetimes = report['lifetimes']
    candidates = report['reuse_candidates']

    print(f"\n  {'节点名':<22} {'生存区间 [first, last]'}")
    print(f"  {'-'*42}")
    for name, lt in lifetimes.items():
        bar_start = lt['first']
        bar_end = lt['last']
        bar = ' ' * bar_start + '█' * (bar_end - bar_start + 1)
        print(f"  {name:<22} [{bar_start:>2}, {bar_end:>2}]  {bar}")

    print(f"\n  可共享内存的节点对（生命周期不重叠）:")
    if not candidates:
        print("    （无）")
    else:
        for a_name, b_name, a_end, b_start in candidates:
            print(f"    {a_name} (结束@{a_end})  ↔  {b_name} (开始@{b_start})")
    print(f"\n  共发现 {len(candidates)} 对可复用内存候选")


# ============================================================
# 主流程：逐个演示每个 Pass
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    # ---------- Pass 1: 常量折叠 ----------
    print("\n" + "▶" * 3 + "  Pass 1：常量折叠（Constant Folding）")
    model_cf = NormModel().eval()
    traced_cf = fx.symbolic_trace(model_cf)
    print_graph(traced_cf, "常量折叠 — 优化前")
    traced_cf_opt = pass_constant_folding(traced_cf)
    print_graph(traced_cf_opt, "常量折叠 — 优化后")
    # 验证数值一致性
    x_cf = torch.randn(4, 3)
    with torch.no_grad():
        out_before = model_cf(x_cf)
        out_after  = traced_cf_opt(x_cf)
    print(f"  数值一致 (allclose): {torch.allclose(out_before, out_after, atol=1e-5)} ✅")

    # ---------- Pass 2: 代数简化 ----------
    print("\n" + "▶" * 3 + "  Pass 2：代数简化与算子替换（Algebraic Simplification）")
    model_as = AlgebraModel().eval()
    traced_as = fx.symbolic_trace(model_as)
    print_graph(traced_as, "代数简化 — 优化前")
    traced_as_opt = pass_algebraic_simplification(traced_as)
    print_graph(traced_as_opt, "代数简化 — 优化后")
    x_as = torch.randn(8, 64)
    with torch.no_grad():
        out_before = model_as(x_as)
        out_after  = traced_as_opt(x_as)
    print(f"  数值一致 (allclose): {torch.allclose(out_before, out_after, atol=1e-5)} ✅")

    # ---------- Pass 3: 公共子表达式消除 ----------
    print("\n" + "▶" * 3 + "  Pass 3：公共子表达式消除（CSE）")
    model_cse = CSEModel().eval()
    traced_cse = fx.symbolic_trace(model_cse)
    print_graph(traced_cse, "CSE — 优化前")
    traced_cse_opt = pass_cse(traced_cse)
    print_graph(traced_cse_opt, "CSE — 优化后")
    x_cse = torch.randn(4, 64)
    with torch.no_grad():
        out_before = model_cse(x_cse)
        out_after  = traced_cse_opt(x_cse)
    print(f"  数值一致 (allclose): {torch.allclose(out_before, out_after, atol=1e-5)} ✅")

    # ---------- Pass 4: 静态内存规划 ----------
    print("\n" + "▶" * 3 + "  Pass 4：静态内存规划（Static Memory Planning）")
    model_mp = MemPlanModel().eval()
    traced_mp = fx.symbolic_trace(model_mp)
    print_graph(traced_mp, "静态内存规划 — 计算图")
    report = pass_static_memory_planning(traced_mp)
    print("\n  生命周期分析：")
    print_memory_plan(report)

