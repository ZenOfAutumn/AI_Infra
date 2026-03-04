"""
torch.compile 功能演示

按照 TorchCompile.md 文档章节顺序，逐一演示：
  Demo 1  — 基础用法：一行编译，推理加速（§3.1）
  Demo 2  — 编译模式对比：default / reduce-overhead（§3.2）
  Demo 3  — fullgraph 参数：强制完整编译与 Graph Break 检测（§3.3 / §6.1）
  Demo 4  — dynamic 参数：动态形状支持与 recompile 计数（§3.4）
  Demo 5  — 禁用编译：torch.compiler.disable() 调试用法（§3.5）
  Demo 6  — 训练场景：前向 + 反向均加速（§4）
  Demo 7  — Guard 机制：形状变化触发 recompile（§2.1）
  Demo 8  — 自定义后端：接管 FX 图打印（§7.2）
  Demo 9  — 捕获 FX 图：通过自定义后端查看 Dynamo 提取的计算图（§7.2）
"""

import platform
import time

import torch
import torch.nn as nn

# macOS 上 TorchInductor 需要 libc++.1.dylib，可能因系统 rpath 问题无法加载；
# 此时自动降级到 aot_eager（图捕获 + AOTAutograd，跳过 C++ 内核生成），
# 在 Linux / GPU 环境下直接使用默认的 inductor 后端。
_ON_MACOS = platform.system() == 'Darwin'
_DEFAULT_BACKEND = 'aot_eager' if _ON_MACOS else 'inductor'
print(f"[环境] 平台: {platform.system()}, 编译后端: {_DEFAULT_BACKEND}")


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def benchmark(fn, *args, n_warmup: int = 3, n_run: int = 20) -> float:
    """测量函数执行的平均耗时（毫秒）。"""
    with torch.no_grad():
        for _ in range(n_warmup):
            fn(*args)
        t0 = time.perf_counter()
        for _ in range(n_run):
            fn(*args)
        t1 = time.perf_counter()
    return (t1 - t0) / n_run * 1000


def section(title: str):
    print(f"\n{'━' * 60}")
    print(f"  {title}")
    print(f"{'━' * 60}")


# ─────────────────────────────────────────────────────────────
# 共用模型
# ─────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """三层 MLP，含 ReLU 激活，用于大多数 demo。"""
    def __init__(self, in_dim=128, hidden=512, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# Demo 1：基础用法（§3.1）
# ─────────────────────────────────────────────────────────────

def demo_basic():
    section("Demo 1 — 基础用法：一行 torch.compile 推理加速")

    model = MLP().eval()
    x = torch.randn(64, 128)

    # ① 原始 eager 模式
    t_eager = benchmark(model, x)

    # ② torch.compile（首次调用含编译时间，不计入测量）
    torch._dynamo.reset()
    compiled = torch.compile(model, backend=_DEFAULT_BACKEND)
    compiled(x)                              # warm-up / 触发编译
    t_compile = benchmark(compiled, x)

    # 验证数值一致
    with torch.no_grad():
        out_eager  = model(x)
        out_compile = compiled(x)
    match = torch.allclose(out_eager, out_compile, atol=1e-5)

    print(f"  Eager  推理耗时: {t_eager:.4f} ms")
    print(f"  Compile推理耗时: {t_compile:.4f} ms")
    print(f"  加速比          : {t_eager / t_compile:.2f}x")
    print(f"  数值一致 (allclose): {match} ✅")


# ─────────────────────────────────────────────────────────────
# Demo 2：编译模式对比（§3.2）
# ─────────────────────────────────────────────────────────────

def demo_modes():
    section("Demo 2 — 编译模式对比：default vs reduce-overhead")

    model = MLP().eval()
    x = torch.randn(64, 128)

    t_eager = benchmark(model, x)

    # macOS 上 mode 参数在 aot_eager 后端下无效，统一用 aot_eager 对比
    # Linux/GPU 环境下使用真实的 inductor 模式
    modes = ['aot_eager', 'aot_eager'] if _ON_MACOS else ['default', 'reduce-overhead']
    mode_labels = ['aot_eager (run1)', 'aot_eager (run2)'] if _ON_MACOS else ['default', 'reduce-overhead']

    results = {}
    for label, mode_or_backend in zip(mode_labels, modes):
        torch._dynamo.reset()
        if _ON_MACOS:
            c = torch.compile(model, backend=mode_or_backend)
        else:
            c = torch.compile(model, mode=mode_or_backend)
        c(x)                                 # 触发编译
        results[label] = benchmark(c, x)

    hint = '(macOS: aot_eager 后端，GPU 环境下可见 default/reduce-overhead 差异)' if _ON_MACOS else ''
    print(f"  {'模式/后端':<25} {'耗时 (ms)':>12} {'vs Eager':>10}  {hint}")
    print(f"  {'-'*55}")
    print(f"  {'eager':<25} {t_eager:>12.4f} {'—':>10}")
    for label, t in results.items():
        speedup = t_eager / t
        print(f"  {label:<25} {t:>12.4f} {speedup:>9.2f}x")


# ─────────────────────────────────────────────────────────────
# Demo 3：fullgraph 参数 & Graph Break 检测（§3.3 / §6.1）
# ─────────────────────────────────────────────────────────────

class ModelWithGraphBreak(nn.Module):
    """刻意在 forward 里加入会导致 Graph Break 的 Python 操作。"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        # ← Graph Break：依赖 tensor 数值的 Python if 分支
        if x.sum().item() > 0:
            x = torch.relu(x)
        else:
            x = torch.tanh(x)
        return x


def demo_fullgraph():
    section("Demo 3 — fullgraph 参数 & Graph Break 检测")

    model_break = ModelWithGraphBreak().eval()
    x = torch.randn(8, 64)

    # ① 默认模式：Graph Break 会静默 Fallback，不报错
    torch._dynamo.reset()
    compiled_default = torch.compile(model_break, backend=_DEFAULT_BACKEND)
    out1 = compiled_default(x)
    print(f"  [default]  编译成功（含 Graph Break，静默 Fallback），输出形状: {tuple(out1.shape)}")

    # ② fullgraph=True：Graph Break 直接抛出错误，用于生产前验证
    torch._dynamo.reset()
    compiled_full = torch.compile(model_break, backend=_DEFAULT_BACKEND, fullgraph=True)
    try:
        out2 = compiled_full(x)
        print(f"  [fullgraph] 意外成功，输出形状: {tuple(out2.shape)}")
    except Exception as e:
        print(f"  [fullgraph] ✅ 预期错误（Graph Break 被检测到）:")
        print(f"    {type(e).__name__}: {str(e)[:120]}...")

    # ③ 干净模型使用 fullgraph=True 正常通过
    torch._dynamo.reset()
    clean_model = MLP().eval()
    compiled_clean = torch.compile(clean_model, backend=_DEFAULT_BACKEND, fullgraph=True)
    out3 = compiled_clean(torch.randn(8, 128))
    print(f"  [fullgraph + 干净模型] ✅ 编译成功，输出形状: {tuple(out3.shape)}")


# ─────────────────────────────────────────────────────────────
# Demo 4：dynamic 参数（§3.4）
# ─────────────────────────────────────────────────────────────

def demo_dynamic():
    section("Demo 4 — dynamic 参数：动态形状支持")

    model = MLP().eval()

    # dynamic=False（默认）：每个新 shape 触发 recompile
    torch._dynamo.reset()
    compiled_static = torch.compile(model, backend=_DEFAULT_BACKEND, dynamic=False)

    shapes = [(8, 128), (16, 128), (32, 128)]
    print("  [dynamic=False] 不同形状的首次推理（含编译）耗时：")
    for bs, dim in shapes:
        x = torch.randn(bs, dim)
        t0 = time.perf_counter()
        compiled_static(x)
        t1 = time.perf_counter()
        print(f"    batch={bs:>3}: {(t1 - t0) * 1000:.1f} ms（首次含编译）")

    # dynamic=True：符号形状，一次编译适配多种尺寸
    torch._dynamo.reset()
    compiled_dynamic = torch.compile(model, backend=_DEFAULT_BACKEND, dynamic=True)

    print("\n  [dynamic=True]  不同形状的首次推理（含编译）耗时：")
    for bs, dim in shapes:
        x = torch.randn(bs, dim)
        t0 = time.perf_counter()
        compiled_dynamic(x)
        t1 = time.perf_counter()
        print(f"    batch={bs:>3}: {(t1 - t0) * 1000:.1f} ms")

    # 验证两者数值一致
    x_test = torch.randn(4, 128)
    with torch.no_grad():
        out_static  = compiled_static(x_test)
        out_dynamic = compiled_dynamic(x_test)
    print(f"\n  static vs dynamic 数值一致: {torch.allclose(out_static, out_dynamic, atol=1e-5)} ✅")


# ─────────────────────────────────────────────────────────────
# Demo 5：禁用编译（§3.5）
# ─────────────────────────────────────────────────────────────

def demo_disable():
    section("Demo 5 — 禁用编译：@torch._dynamo.disable 装饰器调试用法")

    model = MLP().eval()
    x = torch.randn(16, 128)

    # ① 编译版 forward
    torch._dynamo.reset()
    compiled = torch.compile(model, backend=_DEFAULT_BACKEND)
    compiled(x)  # 触发编译
    t_compiled = benchmark(compiled, x)

    # ② 用 @torch._dynamo.disable 装饰一个函数，强制它在 eager 模式运行。
    #    这是"在部分代码路径上禁用 compile"的官方推荐方式，
    #    适用于调试时想绕过特定子模块的编译。
    @torch._dynamo.disable
    def eager_forward(inp):
        return model(inp)

    eager_forward(x)  # 预热
    t_disabled = benchmark(eager_forward, x)

    print(f"  编译模式耗时        : {t_compiled:.4f} ms")
    print(f"  @disable 禁用后耗时 : {t_disabled:.4f} ms（等同 eager 模式）")

    # 验证数值一致
    with torch.no_grad():
        out_compiled = compiled(x)
        out_disabled = eager_forward(x)
    print(f"  数值一致 (allclose) : {torch.allclose(out_compiled, out_disabled, atol=1e-5)} ✅")
    print("  说明：在 eager 模式下直接调用原始 model，两者结果相同。")


# ─────────────────────────────────────────────────────────────
# Demo 6：训练场景（§4）
# ─────────────────────────────────────────────────────────────

def demo_training():
    section("Demo 6 — 训练场景：前向 + 反向均加速")

    model_eager   = MLP().train()
    model_compile = MLP().train()
    # 让两个模型参数一致，方便对比
    model_compile.load_state_dict(model_eager.state_dict())

    optimizer_eager   = torch.optim.AdamW(model_eager.parameters(),   lr=1e-3)
    optimizer_compile = torch.optim.AdamW(model_compile.parameters(), lr=1e-3)

    compiled = torch.compile(model_compile, backend=_DEFAULT_BACKEND)

    x = torch.randn(64, 128)
    y = torch.randn(64, 64)

    def train_step_eager():
        optimizer_eager.zero_grad()
        loss = ((model_eager(x) - y) ** 2).mean()
        loss.backward()
        optimizer_eager.step()
        return loss.item()

    def train_step_compile():
        optimizer_compile.zero_grad()
        loss = ((compiled(x) - y) ** 2).mean()
        loss.backward()
        optimizer_compile.step()
        return loss.item()

    # 预热（首次含编译）
    for _ in range(3):
        train_step_eager()
        train_step_compile()

    # 计时
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        train_step_eager()
    t_eager = (time.perf_counter() - t0) / n * 1000

    t0 = time.perf_counter()
    for _ in range(n):
        train_step_compile()
    t_compile = (time.perf_counter() - t0) / n * 1000

    print(f"  Eager  训练 step 耗时: {t_eager:.4f} ms")
    print(f"  Compile训练 step 耗时: {t_compile:.4f} ms")
    print(f"  训练加速比            : {t_eager / t_compile:.2f}x")


# ─────────────────────────────────────────────────────────────
# Demo 7：Guard 机制 — 形状变化触发 recompile（§2.1）
# ─────────────────────────────────────────────────────────────

def demo_guard():
    section("Demo 7 — Guard 机制：形状变化触发 recompile")

    model = MLP().eval()
    recompile_count = [0]

    # 注入自定义后端来统计编译次数
    def counting_backend(gm, example_inputs):
        recompile_count[0] += 1
        print(f"    ↻ 第 {recompile_count[0]} 次编译，输入形状: "
              f"{[tuple(t.shape) for t in example_inputs if hasattr(t, 'shape')]}")
        return gm.forward

    torch._dynamo.reset()
    compiled = torch.compile(model, backend=counting_backend)

    print("  调用序列：")
    inputs = [
        (8,  128),   # 首次：触发编译
        (8,  128),   # 同形状：Guard 通过，复用
        (8,  128),   # 同形状：Guard 通过，复用
        (16, 128),   # 新形状：Guard 失败，重新编译
        (16, 128),   # 同形状：Guard 通过，复用
        (32, 128),   # 新形状：Guard 失败，重新编译
    ]
    for i, (bs, dim) in enumerate(inputs):
        x = torch.randn(bs, dim)
        compiled(x)
        print(f"    调用 {i+1}: shape=[{bs},{dim}]  → 累计编译次数: {recompile_count[0]}")


# ─────────────────────────────────────────────────────────────
# Demo 8：自定义后端 — 查看 Dynamo 捕获的 FX 图（§7.2）
# ─────────────────────────────────────────────────────────────

def demo_custom_backend():
    section("Demo 8 — 自定义后端：打印 Dynamo 捕获的 FX 图")

    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    ).eval()

    captured_graphs = []

    def inspect_backend(gm: torch.fx.GraphModule, example_inputs):
        """自定义后端：打印 FX 图，然后返回原始 forward 不做额外优化。"""
        captured_graphs.append(gm)
        print(f"\n  === Dynamo 捕获到的 FX 图（共 {len(list(gm.graph.nodes))} 个节点）===")
        print(f"  {'节点名':<25} {'op':<16} {'target'}")
        print(f"  {'-'*58}")
        for node in gm.graph.nodes:
            print(f"  {node.name:<25} {node.op:<16} {node.target}")
        return gm.forward

    torch._dynamo.reset()
    compiled = torch.compile(model, backend=inspect_backend)

    x = torch.randn(4, 16)
    compiled(x)   # 触发追踪，inspect_backend 被调用

    print(f"\n  共捕获 {len(captured_graphs)} 个 FX 子图")

    # 验证数值
    with torch.no_grad():
        out_eager   = model(x)
        out_compiled = compiled(x)
    print(f"  数值一致 (allclose): {torch.allclose(out_eager, out_compiled, atol=1e-5)} ✅")


# ─────────────────────────────────────────────────────────────
# Demo 9：查看 Dynamo 的 explain 输出（§6.1）
# ─────────────────────────────────────────────────────────────

def demo_explain():
    section("Demo 9 — torch._dynamo.explain：分析 Graph Break 位置")

    model = ModelWithGraphBreak().eval()
    x = torch.randn(4, 64)

    torch._dynamo.reset()
    explain_output = torch._dynamo.explain(model)(x)

    print(f"  图段数量      : {explain_output.graphs.__len__()}")
    print(f"  Graph Break 数: {len(explain_output.break_reasons)}")
    if explain_output.break_reasons:
        print("  Break 原因列表:")
        for i, reason in enumerate(explain_output.break_reasons):
            print(f"    [{i+1}] {str(reason.reason)[:100]}")


# ─────────────────────────────────────────────────────────────
# 主程序：依次运行所有 Demo
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'█' * 60}")
    print(f"  torch.compile Demo — PyTorch {torch.__version__}")
    print(f"  设备: {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print(f"{'█' * 60}")

    demo_basic()
    demo_modes()
    demo_fullgraph()
    demo_dynamic()
    demo_disable()
    demo_training()
    demo_guard()
    demo_custom_backend()
    demo_explain()

    print(f"\n{'█' * 60}")
    print("  所有 Demo 运行完成 ✅")
    print(f"{'█' * 60}\n")

