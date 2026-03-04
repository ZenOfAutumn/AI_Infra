# torch.compile 原理与使用说明

> PyTorch 2.0 核心特性，通过一行代码实现训练/推理的端到端自动加速。

---

## 1. 什么是 torch.compile？

`torch.compile` 是 PyTorch 2.0 引入的**即时编译（JIT Compilation）框架**，目标是在完全兼容 Python 动态语义的前提下，将 `nn.Module` 的执行路径自动编译为底层高效内核，无需手动修改模型代码。

```python
model = MyModel()
model = torch.compile(model)   # 一行代码，推理/训练均可加速
output = model(input)
```

与 TorchScript / Torch FX 的核心区别：

| | torch.compile | TorchScript | Torch FX |
|:---|:---|:---|:---|
| **使用门槛** | 一行装饰器，零改动 | 需类型注解，受语法限制 | 需手动编写 Pass |
| **控制流支持** | ✅ 完整（含数据依赖分支）| ✅ 完整 | ❌ 有限 |
| **目标** | 训练+推理自动加速 | 跨平台/C++部署 | 图变换工具链 |
| **后端** | Triton / CUDA / CPU | LibTorch C++ | Python |
| **加速原理** | 算子融合 + 内核生成 | 消除 Python GIL | 图结构变换 |

---

## 2. 整体架构：三层流水线

```
Python nn.Module
      │
      ▼  ① TorchDynamo（图捕获前端）
      │     拦截 Python 字节码，提取可编译子图（FX Graph）
      │
      ▼  ② AOTAutograd（自动微分后端）
      │     将前向图 + 反向图一起提前（Ahead-of-Time）编译
      │
      ▼  ③ TorchInductor（代码生成后端）
            将 FX 图编译为：
            ├── GPU：Triton 内核（算子融合 + 向量化）
            └── CPU：C++ / OpenMP 代码
```

### 2.1 TorchDynamo（图捕获）

**核心机制**：在 CPython 解释器层面注入一个"帧评估钩子"（`PEP 523`），拦截函数的字节码执行，用**符号追踪**的方式提取出可以静态分析的计算子图（Guard-protected subgraph）。

- 对于"纯张量计算"部分 → 提取为 FX 图，交给后端编译
- 对于无法追踪的 Python 动态特性（print、条件分支依赖 Python 对象等）→ **回退（Fallback）**到普通 Python 执行，不会出错

**Guard 机制**：每次执行时检查输入的形状/类型/值是否与上次编译时一致。若一致则复用已编译内核；若不一致则**重新追踪编译（Recompile）**。

```
第1次调用: 追踪 → 生成内核 K1（针对 shape=[8,128]）
第2次调用: shape=[8,128] → Guard通过 → 直接执行 K1（快）
第3次调用: shape=[16,128] → Guard失败 → 重新追踪 → 生成 K2（慢）
```

### 2.2 AOTAutograd（提前自动微分）

**核心机制**：将前向计算图和反向梯度计算图**在编译期**同时展开（而非运行时动态构建），使得整个训练迭代（前向+反向）可以作为一个**完整的静态图**交给后端优化。

```
动态模式（eager）：
  forward() → 运行时构建 autograd graph → backward()
  [每个 step 都重新构建，overhead 大]

AOTAutograd：
  编译期展开 forward+backward → 合并为单一静态图
  [运行时直接执行已编译内核，overhead 极小]
```

### 2.3 TorchInductor（代码生成）

**默认后端**，将 FX 图转换为高性能代码：

- **GPU 路径**：生成 [Triton](https://github.com/openai/triton) 内核，自动完成：
  - **算子融合**（pointwise + reduction 合并为单个 kernel）
  - **内存布局优化**（tiling、向量化）
  - **共享显存复用**（减少 HBM 带宽压力）
- **CPU 路径**：生成 C++ 代码，利用 OpenMP 并行和 SIMD 向量化

---

## 3. 使用方法

### 3.1 基础用法

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
)
model.eval()

# 编译（首次调用时触发追踪+编译，后续调用直接执行编译好的内核）
compiled_model = torch.compile(model)

x = torch.randn(32, 128)
with torch.no_grad():
    out = compiled_model(x)
```

### 3.2 编译模式（`mode` 参数）

`torch.compile` 提供三种预设模式，在**编译速度**和**运行时加速比**之间权衡：

| 模式 | 说明 | 适用场景 |
| :--- | :--- | :--- |
| `"default"` | 默认模式，平衡编译时间与运行性能 | 通用场景 |
| `"reduce-overhead"` | 激进优化，使用 CUDA Graphs 减少 kernel launch 开销 | 推理，batch 固定 |
| `"max-autotune"` | 极限优化，穷举搜索最优 tiling 策略（编译慢，运行最快）| 高吞吐量推理/训练 |

```python
# 推理场景：优先用 reduce-overhead
compiled = torch.compile(model, mode="reduce-overhead")

# 追求极限吞吐量（接受较长编译时间）
compiled = torch.compile(model, mode="max-autotune")
```

### 3.3 `fullgraph` 参数（禁止 Fallback）

默认情况下，Dynamo 遇到无法追踪的代码会 Fallback 到 Python。设置 `fullgraph=True` 可以**强制要求整个模型必须完整编译**，Fallback 时直接报错——适合用于验证模型是否完全可编译：

```python
# 确保模型没有任何 Python fallback（生产部署前验证用）
compiled = torch.compile(model, fullgraph=True)
```

### 3.4 `dynamic` 参数（动态形状支持）

```python
# 支持动态形状（适合 batch size / seq_len 可变的场景）
compiled = torch.compile(model, dynamic=True)
```

| `dynamic` | 行为 |
| :--- | :--- |
| `False`（默认）| 对每个新形状重新编译，性能最优但 recompile 次数多 |
| `True` | 使用符号形状（`symint`），编译一次可处理多种尺寸 |
| `None` | 自动检测：首次遇到形状变化时自动升级为动态模式 |

### 3.5 禁用编译（调试用）

```python
# 临时禁用 compile，回退到 eager 模式调试
with torch.compiler.disable():
    out = compiled_model(x)

# 或通过环境变量全局禁用
# TORCH_COMPILE_DISABLE=1 python train.py
```

---

## 4. 训练场景使用

`torch.compile` 同样支持训练，前向和反向均会被加速：

```python
model = MyTransformer().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 编译整个模型（含反向传播）
compiled_model = torch.compile(model)

for batch in dataloader:
    optimizer.zero_grad()
    loss = compiled_model(batch).loss
    loss.backward()      # 反向图也在编译范围内
    optimizer.step()
```

> **实测加速比**（来自 PyTorch 官方基准）：
> - LLaMA-7B 训练吞吐量提升约 **1.4x～1.7x**
> - ResNet-50 训练提升约 **1.3x**
> - Stable Diffusion 推理提升约 **2.5x**（GPU，`reduce-overhead` 模式）

---

## 5. 与 CUDA Graphs 的结合（`reduce-overhead` 模式原理）

`mode="reduce-overhead"` 内部会尝试将多个 kernel launch 合并为一个 **CUDA Graph**：

```
普通执行（每个算子单独 launch）：
  CPU: launch kernel_1 → launch kernel_2 → launch kernel_3 ...
  GPU: [kernel_1]──────[kernel_2]──────[kernel_3]
       ↑每次都有 CPU→GPU 调度延迟（~5-10μs/次）

CUDA Graph（批量录制后整体 replay）：
  录制: replay graph_A（包含 kernel_1,2,3...）
  执行: CPU 发一次命令 → GPU 连续执行所有 kernel（无调度延迟）
```

适用条件：输入形状固定，无动态控制流（推理场景最理想）。

---

## 6. 常见问题与限制

### 6.1 Graph Break（图断裂）

当 Dynamo 遇到无法追踪的代码时，会在该处"断开"图，分成多段分别编译。断裂点越多，编译收益越小。

**常见导致 Graph Break 的操作**：
- `print()` / `logging` 调用
- Python 原生的 `list`/`dict` 操作（非 Tensor）
- 依赖数据值的 `if tensor.item() > 0`
- 调用未被支持的第三方库函数

**诊断 Graph Break**：

```python
# 打印所有 graph break 的位置和原因
import torch._dynamo
torch._dynamo.explain(model)(x)

# 或设置环境变量
# TORCH_LOGS=graph_breaks python script.py
```

### 6.2 首次编译延迟（Warm-up）

第一次调用 `compiled_model(x)` 会触发追踪和内核编译，耗时可能达到**数十秒**（`max-autotune` 模式可能数分钟）。

**缓存编译结果**（避免每次重启都重新编译）：

```python
# 启用磁盘缓存（PyTorch 2.4+）
import torch._inductor.config as inductor_config
inductor_config.fx_graph_cache = True

# 或通过环境变量
# TORCHINDUCTOR_CACHE_DIR=/tmp/torch_compile_cache python script.py
```

### 6.3 不支持的操作

目前不支持或支持有限的场景：
- 模型内部调用 `torch.save` / `torch.load`
- 含 Python 生成器（generator）的 forward
- 部分自定义 C++ 扩展（需要 `allow_in_graph` 标注）

---

## 7. 调试与可观测性

### 7.1 查看生成的 Triton 内核

```python
import os
os.environ["TORCH_COMPILE_DEBUG"] = "1"

compiled_model = torch.compile(model)
out = compiled_model(x)
# 在 /tmp/torchinductor_xxx/ 目录下查看生成的 .py（Triton）文件
```

### 7.2 查看捕获的 FX 图

```python
# 打印 Dynamo 捕获的 FX 图
torch._dynamo.reset()

def my_backend(gm: torch.fx.GraphModule, example_inputs):
    print("=== 捕获到的 FX 图 ===")
    gm.print_readable()   # 打印图的可读形式
    return gm.forward     # 直接返回，不做额外优化

compiled = torch.compile(model, backend=my_backend)
compiled(x)
```

### 7.3 使用自定义后端

```python
# 可以接入自己的 FX 优化 Pass 作为 compile 后端
def my_custom_backend(gm: torch.fx.GraphModule, example_inputs):
    # 在这里插入自定义的 FX Pass（如量化、剪枝）
    gm = my_fx_pass(gm)
    return torch.jit.script(gm).forward

compiled = torch.compile(model, backend=my_custom_backend)
```

---

## 8. 在 LLM 推理链路中的位置

```
LLM 推理请求
      │
      ▼  vLLM / TGI 调度层（动态批处理）
      │
      ▼  torch.compile（prefill / decode 阶段静态子图编译）
      │      ├── TorchDynamo 捕获 Attention / FFN 计算图
      │      ├── AOTAutograd 提前展开（推理时无需 backward）
      │      └── TorchInductor 生成融合 Triton 内核
      │
      ▼  CUDA Graph replay（固定形状的 decode 步骤）
      │
      ▼  输出 Token
```

**LLaMA / Mistral 等模型的典型优化链路**：
1. `torch.compile(model, mode="reduce-overhead")` — 融合 Attention + RMSNorm + FFN
2. FlashAttention 作为算子替换（通过 `torch.nn.functional.scaled_dot_product_attention`）
3. CUDA Graphs 消除 decode 阶段逐 token 的 kernel launch 开销

---

## 9. torch.compile vs TorchScript vs Torch FX 选型建议

| 需求 | 推荐方案 | 原因 |
| :--- | :--- | :--- |
| 训练加速，代码零改动 | **torch.compile** | 一行代码，兼容动态图语义 |
| 推理加速，形状固定 | **torch.compile** + `reduce-overhead` | CUDA Graphs 加持，延迟最低 |
| 部署到 C++ / 移动端 | **TorchScript** | 生成可脱离 Python 的 .pt 文件 |
| 量化 / 剪枝 / 自定义图变换 | **Torch FX** | 细粒度图操作，可与 compile 联动 |
| 极限性能调优（手写 kernel） | **Triton** / CUDA C++ | 完全控制内存访问和并行策略 |

---

## 10. 参考资料

- [PyTorch 2.0 官方博客](https://pytorch.org/blog/pytorch-2.0-release/)
- [TorchDynamo 设计文档](https://pytorch.org/docs/stable/dynamo/index.html)
- [TorchInductor 介绍](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
- [torch.compile 使用指南](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)

