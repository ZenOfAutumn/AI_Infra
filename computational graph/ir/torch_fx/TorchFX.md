# Torch FX 原理与使用说明

## 1. 什么是 Torch FX？

Torch FX（`torch.fx`）是 PyTorch 提供的一套**Python 层面的计算图捕获与变换框架**，于 PyTorch 1.8 正式引入。它的核心价值在于：在不离开 Python 的前提下，对 `nn.Module` 的计算图进行**程序化的读取、修改和重新编译**。

```
nn.Module (Python 动态图)
        │
        ▼  fx.symbolic_trace()
        │
  fx.Graph (静态 IR，可遍历/修改)
        │
        ▼  GraphModule.recompile()
        │
优化后的 nn.Module（可直接执行）
```

与 TorchScript 相比，Torch FX 是**完全在 Python 侧工作**的——它不需要编译成 C++ IR，也不需要类型注解，更适合做**训练时优化和工具链开发**。

---

## 2. 核心架构：三层抽象

### 2.1 `Graph`（计算图）

`fx.Graph` 是一个有序的节点列表（DAG），描述了模型 `forward()` 的完整执行流程。每个节点（`fx.Node`）包含以下关键属性：

| 属性 | 含义 |
| :--- | :--- |
| `name` | 节点的唯一名称（字符串） |
| `op` | 操作类型（见下表） |
| `target` | 操作目标（函数/方法/模块名） |
| `args` | 位置参数（来自其他节点或常量） |
| `kwargs` | 关键字参数 |
| `users` | 使用该节点输出的所有后续节点 |

节点操作类型（`op`）一共有 6 种：

| `op` 值 | 含义 | 示例 |
| :--- | :--- | :--- |
| `placeholder` | 函数输入参数 | 模型的 `x` 输入 |
| `get_attr` | 读取模型属性（如权重） | `self.weight` |
| `call_function` | 调用全局函数 | `torch.relu(x)` |
| `call_method` | 调用 Tensor 方法 | `x.view(...)` |
| `call_module` | 调用子模块的 `forward` | `self.linear1(x)` |
| `output` | 图的输出节点 | `return x` |

### 2.2 `GraphModule`（可执行的图模块）

`fx.GraphModule` 是一个标准的 `nn.Module`，内部持有一个 `fx.Graph`。调用 `recompile()` 后，FX 会根据图结构**动态生成 Python 代码**，使其可以像普通模块一样运行。

```python
traced = fx.symbolic_trace(model)
print(traced.code)  # 查看 FX 生成的 Python 代码
```

### 2.3 `Interpreter` 与 `Transformer`（图遍历工具）

- **`fx.Interpreter`**：逐节点执行图，可在每个节点前后插入自定义逻辑（如 profiling、量化校准）。
- **`fx.Transformer`**：继承并重写节点变换逻辑，用于批量修改图结构。

---

## 3. 核心 API：捕获计算图

### 3.1 `fx.symbolic_trace`

```python
import torch.fx as fx

traced: fx.GraphModule = fx.symbolic_trace(model)
```

**原理**：`symbolic_trace` 用**符号值（`fx.Proxy`）**替代真实张量，对模型执行一次"干跑"（dry run）。每次 Python 操作都被 `Proxy` 拦截并记录为图节点，最终生成完整的 `Graph`。

**限制**：
- 不支持**数据依赖的控制流**（如 `if x.sum() > 0`），因为符号值没有真实数值
- 不支持 Python 原生的动态特性（如 `*args` 动态长度）

### 3.2 处理控制流：`concrete_args`

对于含有静态条件的模型，可以用 `concrete_args` 固定参数值，使 FX 能追踪特定的执行路径：

```python
# 固定 training=False 路径进行追踪
traced = fx.symbolic_trace(model, concrete_args={'training': False})
```

---

## 4. 图优化 Pass

FX 的核心使用模式是编写**图变换 Pass**，每个 Pass 接收一个 `GraphModule`，对图进行修改后返回新的 `GraphModule`。

### 4.1 Pass 基本模式

```python
def my_pass(gm: fx.GraphModule) -> fx.GraphModule:
    for node in gm.graph.nodes:
        # 分析节点，按需修改
        if should_transform(node):
            with gm.graph.inserting_before(node):
                new_node = gm.graph.call_function(...)
            node.replace_all_uses_with(new_node)
            gm.graph.erase_node(node)
    gm.recompile()
    return gm
```

### 4.2 Pass 1：死代码消除（Dead Code Elimination, DCE）

**目标**：删除输出未被任何节点使用的"无用"节点，减少不必要的计算。

```python
def pass_dead_code_elimination(gm: fx.GraphModule) -> fx.GraphModule:
    gm.graph.eliminate_dead_code()   # FX 内置 DCE
    gm.recompile()
    return gm
```

典型场景：`eval` 模式下 `Dropout(p=0.0)` 是恒等变换，但 FX 仍会为其创建节点，DCE 可将其消除。

**优化前**（7 个节点）：
```
x → linear1 → relu1 → dropout → linear2 → relu2 → output
```

**优化后**（6 个节点，dropout 被消除）：
```
x → linear1 → relu1 → linear2 → relu2 → output
```

### 4.3 Pass 2：算子融合（Operator Fusion）

**目标**：将相邻的 `Linear → ReLU` 模式合并为单一的 `FusedLinearReLU` 模块。

**融合的意义**（显存带宽角度）：
```
未融合：Linear 写显存 → ReLU 读显存 → ReLU 写显存  （2次显存读写）
已融合：Linear + ReLU 在寄存器中一次性完成         （1次显存写入）
```

**实现步骤**：

```python
def pass_fuse_linear_relu(gm: fx.GraphModule) -> fx.GraphModule:
    for node in list(gm.graph.nodes):
        # 1. 找到 ReLU 节点
        if node.op != 'call_module' or not isinstance(get_module(node), nn.ReLU):
            continue
        prev_node = node.args[0]
        # 2. 检查前驱节点是否为 Linear
        if not isinstance(get_module(prev_node), nn.Linear):
            continue
        # 3. 插入融合节点，替换原始节点
        fused = FusedLinearReLU(get_module(prev_node))
        gm.add_module(fused_name, fused)
        with gm.graph.inserting_after(prev_node):
            fused_node = gm.graph.call_module(fused_name, args=prev_node.args)
        node.replace_all_uses_with(fused_node)
        gm.graph.erase_node(node)
        gm.graph.erase_node(prev_node)
    gm.recompile()
    return gm
```

**优化前**（6 个节点）：
```
x → linear1 → relu1 → linear2 → relu2 → output
```

**优化后**（5 个节点，Linear+ReLU 各融合为一个节点）：
```
x → fused_linear_relu_0 → dropout → fused_linear_relu_1 → output
```

### 4.4 Pass 3：常量折叠（Constant Folding）

**原理**：如果一个节点的所有输入都是常量（来自 `get_attr` 的 buffer/parameter），那么它的输出在推理时永远不会改变。可以在**编译期**直接执行该计算，将结果注册为新的 buffer，用一个 `get_attr` 节点替换掉原来的计算节点，从而消除运行时的重复计算。

**典型场景**：
- 模型中固定的归一化系数（均值 `mean`、标准差 `std`）
- 推理时不变的位置编码（Positional Encoding）
- 预处理中硬编码的 `scale / shift`

**示意图**：
```
优化前: x → [get_attr: mean] → sub → [get_attr: std] → div → output
                                ↑常量                    ↑常量
优化后: x → [get_attr: _folded_0 (预计算结果)] → output
```

**核心实现**：

```python
def pass_constant_folding(gm: fx.GraphModule) -> fx.GraphModule:
    gm = copy.deepcopy(gm)
    # 收集所有 get_attr 节点的真实值
    const_values = {}
    for node in gm.graph.nodes:
        if node.op == 'get_attr':
            val = gm
            for part in node.target.split('.'):
                val = getattr(val, part)
            const_values[node.name] = val

    for node in gm.graph.nodes:
        if node.op not in ('call_function', 'call_method'):
            continue
        # 所有参数都是常量 → 可折叠
        if all((isinstance(a, fx.Node) and a.name in const_values)
               or not isinstance(a, fx.Node) for a in node.args):
            # 提前执行，注册为新 buffer
            result = node.target(*[const_values.get(a.name, a)
                                    if isinstance(a, fx.Node) else a
                                    for a in node.args])
            buf_name = f'_folded_{n}'
            gm.register_buffer(buf_name, result)
            with gm.graph.inserting_before(node):
                new_node = gm.graph.get_attr(buf_name)
            node.replace_all_uses_with(new_node)
            gm.graph.erase_node(node)
    gm.recompile()
    return gm
```

**优化效果**（`NormModel`：`out = (x - mean) / std`）：

```
优化前 5 个节点: x → mean(get_attr) → sub → std(get_attr) → div → output
优化后 3 个节点: x → _folded_0(get_attr) → _folded_1(get_attr) → output
```

---

### 4.5 Pass 4：代数简化与算子替换（Algebraic Simplification）

**原理**：利用数学等价关系，将计算量更大或执行更慢的算子替换为等价的高效形式：

| 原始表达式 | 简化后 | 理由 |
| :--- | :--- | :--- |
| `x * 1` | `x` | 乘 1 恒等，直接消除 |
| `x + 0` | `x` | 加 0 恒等，直接消除 |
| `relu(relu(x))` | `relu(x)` | ReLU 幂等性，外层冗余 |
| `torch.div(x, c)` | `torch.mul(x, 1/c)` | GPU 上乘法比除法快约 2x |
| `x ** 2` | `x * x` | 消除幂运算调用开销 |

**核心实现**（以消除冗余 relu + div→mul 为例）：

```python
def pass_algebraic_simplification(gm: fx.GraphModule) -> fx.GraphModule:
    gm = copy.deepcopy(gm)
    named_modules = dict(gm.named_modules())

    for node in list(gm.graph.nodes):
        # 规则 1：relu(relu(x)) → relu(x)
        if (node.op == 'call_module'
                and isinstance(named_modules.get(node.target), nn.ReLU)):
            prev = node.args[0]
            if (prev.op == 'call_module'
                    and isinstance(named_modules.get(prev.target), nn.ReLU)):
                node.replace_all_uses_with(prev)
                gm.graph.erase_node(node)
                continue

        # 规则 2：torch.div(x, scalar) → torch.mul(x, 1/scalar)
        if (node.op == 'call_function'
                and node.target is torch.div
                and isinstance(node.args[1], (int, float))
                and node.args[1] != 0):
            inv = 1.0 / node.args[1]
            with gm.graph.inserting_before(node):
                new_node = gm.graph.call_function(torch.mul, args=(node.args[0], inv))
            node.replace_all_uses_with(new_node)
            gm.graph.erase_node(node)

    gm.recompile()
    return gm
```

**优化效果**（`AlgebraModel`：含冗余 relu + div 2.0）：

```
优化前 6 个节点: x → linear → relu1 → relu2 → div(2.0) → output
优化后 4 个节点: x → linear → relu1 → mul(0.5) → output
                         ↑ relu2 被消除   ↑ div 变 mul
```

---

### 4.6 Pass 5：公共子表达式消除（CSE, Common Subexpression Elimination）

**原理**：如果图中存在两个"签名完全相同"的节点（相同算子 + 完全相同的输入来源），则只需计算一次，将后续所有引用指向第一次计算的结果，消除冗余计算。

**节点签名** = `(op, target, tuple(输入节点名或常量值))`

**典型场景**：
- Attention 中对同一 `hidden_states` 多次调用相同的 `LayerNorm`
- 多分支网络中各分支对同一输入做相同的 `reshape`/`permute`
- 共享前置特征变换的多任务头

**核心实现**：

```python
def pass_cse(gm: fx.GraphModule) -> fx.GraphModule:
    gm = copy.deepcopy(gm)
    seen = {}   # 签名 → 第一次出现的节点

    for node in list(gm.graph.nodes):
        if node.op in ('placeholder', 'output', 'get_attr'):
            continue
        # 构造签名
        sig = (
            node.op, node.target,
            tuple(('node', a.name) if isinstance(a, fx.Node)
                  else ('const', a) for a in node.args),
        )
        if sig in seen:
            node.replace_all_uses_with(seen[sig])   # 指向第一次的结果
            gm.graph.erase_node(node)
        else:
            seen[sig] = node

    gm.recompile()
    return gm
```

**优化效果**（`CSEModel`：对同一 x 调用两次相同的 `LayerNorm`）：

```
优化前 6 个节点: x → norm(x)[第1次] → linear_a
                   → norm(x)[第2次] → linear_b → add → output

优化后 5 个节点: x → norm(x)[仅1次] → linear_a
                                    → linear_b → add → output
```

---

### 4.7 Pass 6：静态内存规划（Static Memory Planning）

**原理**：在图执行前，分析每个中间张量的**生命周期**（从产生到最后一次被使用），找出生命周期**不重叠**的张量对，标注它们为"可共享内存缓冲区"的候选，从而最小化峰值显存占用。

**生命周期示意**：
```
时间步:    0    1    2    3    4    5    6
fc1:           [████]                        last=2
relu:                [████]                  last=3
fc2:                      [████]             last=4  → fc1 与 fc2 不重叠，可复用显存
relu_1:                        [████]        last=5
fc3:                                [████]   last=6  → relu 与 fc3 不重叠，可复用显存
```

**核心实现**：

```python
def pass_static_memory_planning(gm: fx.GraphModule) -> dict:
    nodes = list(gm.graph.nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
    lifetimes = {}

    for i, node in enumerate(nodes):
        if node.op in ('placeholder', 'output'):
            continue
        last = max((node_index[u] for u in node.users), default=i)
        lifetimes[node.name] = {'first': i, 'last': last}

    # 找出不重叠的候选对
    names = list(lifetimes.keys())
    reuse_candidates = [
        (names[i], names[j])
        for i in range(len(names))
        for j in range(i + 1, len(names))
        if lifetimes[names[i]]['last'] < lifetimes[names[j]]['first']
        or lifetimes[names[j]]['last'] < lifetimes[names[i]]['first']
    ]
    return {'lifetimes': lifetimes, 'reuse_candidates': reuse_candidates}
```

**优化效果**（`MemPlanModel`：三层线性序列）：

```
节点名    生存区间      可复用候选
fc1      [1, 2]   ↔  fc2 [3,4]、relu_1 [4,5]、fc3 [5,6]
relu     [2, 3]   ↔  relu_1 [4,5]、fc3 [5,6]
fc2      [3, 4]   ↔  fc3 [5,6]

→ 共发现 6 对可复用内存候选
  理论上 3 个中间张量只需 2 块内存缓冲区（节省约 33% 峰值显存）
```

> **注意**：此 Pass 输出的是**分析报告**（`dict`），不修改图结构。物理内存的实际复用需结合运行时（如 `torch.compile` 的 `Inductor` 后端或自定义 CUDA 内存池）完成。

---

## 5. 图结构可视化

本项目使用 `matplotlib` 将优化前后的图结构并排展示，生成图像文件：

```
computational graph/ir/fx_graph_compare.png
```

节点颜色含义：

| 颜色 | 节点类型 |
| :--- | :--- |
| 🟢 绿色 | `placeholder`（输入节点） |
| 🔵 蓝色 | `call_module`（子模块调用） |
| 🟠 橙色 | `call_function`（全局函数调用） |
| 🔴 红色 | `output`（输出节点） |
| 🔥 深橙色 | 融合后的算子节点 |

---

## 6. 性能对比结果

以两层 MLP（输入 128 维，隐层 256 维，输出 64 维，batch=512）为例，CPU 上的推理耗时：

| 模型 | 平均耗时 | 说明 |
| :--- | :--- | :--- |
| 原始 `nn.Module` | ~0.35 ms | Python 动态调度 |
| FX Traced（未优化） | ~0.30 ms | FX 捕获后生成静态代码 |
| FX Traced（融合优化） | ~0.28 ms | DCE + 算子融合 |

> **加速比约 1.2x**。在 GPU 上，算子融合对显存带宽利用率的提升效果会更显著（通常可达 1.5x～2x）。

---

## 7. Torch FX vs TorchScript vs torch.compile

| 特性 | Torch FX | TorchScript | torch.compile（2.0） |
| :--- | :--- | :--- | :--- |
| **工作层面** | Python 图 IR | C++ 编译 IR | 多后端（Triton / CUDA）|
| **控制流支持** | ❌ 有限（无数据依赖分支）| ✅ 完整 | ✅ 完整 |
| **主要用途** | 图变换/优化 Pass | 跨语言部署 | 自动内核优化 |
| **可读性** | ✅ 高（生成标准 Python）| ❌ 低（C++ IR）| ✅ 高（透明包装）|
| **入侵性** | 低（无需改源码）| 中（需类型注解）| 极低（一行装饰器）|
| **典型应用** | 量化、剪枝、融合工具链 | 移动端/嵌入式部署 | 训练/推理通用加速 |

**分工建议**：
- **工具链开发**（量化、融合、分析）→ 用 **Torch FX**
- **跨平台/C++ 部署** → 用 **TorchScript**
- **端到端自动加速（无需手写 Pass）** → 用 **torch.compile**

---

## 8. 实际工程使用场景

### 8.1 量化感知训练（QAT）

`torch.quantization` 内部即使用 FX Pass 自动在 `Linear`/`Conv` 前后插入量化/反量化节点：

```python
from torch.quantization import quantize_fx

qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping("qnnpack")
model_prepared = quantize_fx.prepare_fx(model, qconfig_mapping, example_inputs)
```

### 8.2 模型剪枝与结构分析

遍历 FX 图可以精确统计每层的计算量（FLOPs）、参数量，或按策略删除特定节点。完整示例见 `fx_pruning_analysis.py`。

#### 8.2.1 结构分析：统计参数量与 FLOPs

```python
def analyze_graph(gm: fx.GraphModule) -> dict:
    named_modules = dict(gm.named_modules())
    total_params, total_flops = 0, 0
    layer_stats = []

    for node in gm.graph.nodes:
        if node.op != 'call_module':
            continue
        module = named_modules.get(node.target)
        # 参数量
        params = sum(p.numel() for p in module.parameters())
        total_params += params
        # FLOPs（以 Linear 层为例：2 × in_features × out_features）
        flops = 0
        if isinstance(module, nn.Linear):
            flops = 2 * module.in_features * module.out_features
            total_flops += flops
        layer_stats.append({'node': node.name, 'type': type(module).__name__,
                            'params': params, 'flops': flops})
    return {'layer_stats': layer_stats,
            'total_params': total_params, 'total_flops': total_flops}
```

**运行结果示例**（`DeepMLP`，4 个 Linear 层）：

```
节点名          模块类型    参数量     FLOPs
fc1           Linear     33,024    65,536
bottleneck    Linear      8,224    16,384
expand        Linear      8,448    16,384
fc2           Linear     16,448    32,768
合计                      66,144   131,072
```

#### 8.2.2 结构化剪枝 Pass（跳连替换策略）

**核心思路**：删除目标节点后，将其所有下游使用者改为直接使用该节点的上游输入——相当于在图中建立一条跳连（skip-connection），完全绕过被删层。

```python
def pass_prune_layers(gm: fx.GraphModule, prune_targets: set) -> fx.GraphModule:
    gm = copy.deepcopy(gm)
    nodes_to_remove = [
        node for node in gm.graph.nodes
        if node.op == 'call_module' and node.target in prune_targets
    ]
    for node in nodes_to_remove:
        upstream = node.args[0]              # 被删节点的上游输入
        node.replace_all_uses_with(upstream) # 下游直接连到上游（跳连）
    for node in reversed(nodes_to_remove):   # 反向删除，避免引用悬空
        if len(node.users) == 0:
            gm.graph.erase_node(node)
    gm.recompile()
    return gm
```

**使用示例**：删除 `bottleneck → relu_bn → expand → relu2` 瓶颈层链路：

```python
prune_targets = {'bottleneck', 'relu_bn', 'expand', 'relu2'}
traced_pruned = pass_prune_layers(traced, prune_targets)
```

**剪枝前后对比**（`DeepMLP`，batch=256，CPU）：

| 指标 | 剪枝前 | 剪枝后 | 变化 |
| :--- | :--- | :--- | :--- |
| 参数量 | 66,144 | 49,472 | **-25.2%** |
| FLOPs | 131,072 | 98,304 | **-25.0%** |
| 节点数 | 10 | 6 | -4 |
| 推理耗时 | ~0.29 ms | ~0.15 ms | **1.93x 加速** |

> **注意**：结构化剪枝会改变网络的计算路径，直接删除层后精度会下降，实际工程中需配合**知识蒸馏**或**微调**来恢复精度。

### 8.3 编译器后端对接

`torch.compile` 的 `Dynamo` 前端在提取计算图后，正是将其转换为 FX 图格式，再交给 `Inductor`（Triton 后端）生成高效内核。

---

## 9. 参考代码

### `fx_graph_optimization.py`：算子融合与 DCE 优化

- **`MLP`**：包含冗余 `Dropout` 的两层网络（待优化模型）
- **`capture_fx_graph`**：使用 `symbolic_trace` 捕获计算图
- **`pass_dead_code_elimination`**：DCE Pass 实现
- **`pass_fuse_linear_relu`**：算子融合 Pass 实现
- **`benchmark`**：性能测量函数
- **`_visualize_graphs`**：用 `matplotlib` 生成图结构可视化对比图（输出 `fx_graph_compare.png`）

### `fx_pruning_analysis.py`：模型剪枝与结构分析

- **`DeepMLP`**：含瓶颈层的四层 MLP（待剪枝模型）
- **`analyze_graph`**：遍历 FX 图，逐层统计参数量、FLOPs 及节点类型分布
- **`pass_prune_layers`**：结构化剪枝 Pass，通过跳连替换策略删除指定层链路
- **`benchmark`**：推理耗时测量，对比剪枝前后速度

### `fx_passes_extra.py`：进阶优化 Pass（常量折叠 / 代数简化 / CSE / 内存规划）

- **`NormModel` / `pass_constant_folding`**：常量折叠，将 `(x-mean)/std` 的常量部分预计算
- **`AlgebraModel` / `pass_algebraic_simplification`**：消除冗余 relu，将 `div` 替换为 `mul`
- **`CSEModel` / `pass_cse`**：公共子表达式消除，避免重复调用相同 LayerNorm
- **`MemPlanModel` / `pass_static_memory_planning`**：分析张量生命周期，输出可复用内存候选对

