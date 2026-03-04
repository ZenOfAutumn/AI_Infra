# TorchScript 原理与使用说明

## 1. 什么是 TorchScript？

TorchScript 是 PyTorch 提供的一种将动态图模型转换为**静态中间表示（IR）**的机制。它本质上是一个 Python 的子集编译器，能将 Python 代码"冻结"成一个与 Python 解释器完全无关的、可移植的程序。

```
Python 代码 (nn.Module)
        │
        ▼  torch.jit.script 或 torch.jit.trace
        │
TorchScript IR (中间表示)
        │
   ┌────┴─────────────┐
   ▼                  ▼
Python 运行时        C++ 运行时 (LibTorch)
(训练调试)           (生产部署)
```

---

## 2. 核心原理

### 2.1 编译流程

TorchScript 的编译分为两个阶段：

**阶段一：前端解析（Parsing）**

- `torch.jit.script` 模式：直接解析 Python 代码的**抽象语法树（AST）**，分析变量类型，将所有控制流（`if/else/for/while`）一起编译进图中。
- `torch.jit.trace` 模式：给定一组示例输入，**运行一遍**模型，录制执行路径，生成仅包含被执行算子的静态图（不含控制流）。

**阶段二：后端执行（Execution）**

- 生成 TorchScript IR（一种 SSA 形式的低级中间表示）。
- 通过 LibTorch（PyTorch 的 C++ 运行时）执行，完全绕开 Python GIL（全局解释器锁）。

### 2.2 TorchScript IR 示例

对于一个简单的 `y = x + 1; return relu(y)` 操作，TorchScript IR 大致如下：

```
graph(%x : Tensor):
  %1 : int = prim::Constant[value=1]()
  %2 : Tensor = aten::add(%x, %1, %1)   # x + 1
  %3 : Tensor = aten::relu(%2)            # relu(y)
  return (%3)
```

每一行都是一个 SSA（静态单赋值）节点，类型系统完全静态化。

### 2.3 `script` vs `trace` 对比


| 特性         | `torch.jit.script`            | `torch.jit.trace`         |
| :----------- | :---------------------------- | :------------------------ |
| **转换方式** | 解析 Python AST，编译整个函数 | 运行一次，录制操作序列    |
| **控制流**   | ✅ 完整保留`if/for/while`     | ❌ 仅保留被执行的那条路径 |
| **适用场景** | 包含动态控制流的模型          | 纯前馈、无控制流的模型    |
| **类型要求** | 需要显式类型注解              | 从示例输入自动推断        |
| **调试难度** | 编译期报错，信息详细          | 运行期报错，较难追踪      |
| **限制**     | 仅支持 TorchScript 子集语法   | 无法捕获依赖数据的分支    |

---

## 3. 三种核心使用方式

### 3.1 装饰器方式 (`@torch.jit.script`)

直接用装饰器标注一个函数或类，最简单直接：

```python
@torch.jit.script
def relu_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.relu(x + y)
```

### 3.2 `torch.jit.script()` 函数方式

对已有的 `nn.Module` 进行脚本化，**推荐生产部署时使用**：

```python
model = MyModel()
scripted_model = torch.jit.script(model)
scripted_model.save("model.pt")
```

### 3.3 `torch.jit.trace()` 追踪方式

适用于结构固定、不含数据依赖控制流的模型：

```python
example_input = torch.randn(1, 128)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model_traced.pt")
```

### 3.4 混合使用 (`torch.jit.script` + `torch.jit.trace`)

对于复杂模型，可以对子模块分别处理，再组合：

```python
# 主干网络用 trace（结构固定）
backbone = torch.jit.trace(model.backbone, example_input)
# 决策头用 script（包含 if/else）
head = torch.jit.script(model.head)
```

---

## 4. 类型系统：TorchScript 支持的 Python 子集

TorchScript **不是完整的 Python**，它有严格的类型约束：


| 支持类型                      | 说明             |
| :---------------------------- | :--------------- |
| `Tensor`                      | PyTorch 张量     |
| `int`, `float`, `bool`, `str` | 基础标量类型     |
| `List[T]`, `Dict[K, V]`       | 同类型列表和字典 |
| `Optional[T]`                 | 可为 None 的值   |
| `Tuple[T1, T2, ...]`          | 定长元组         |

**不支持的 Python 特性：**

- `*args` / `**kwargs` 动态参数
- Lambda 表达式
- 生成器（generator）
- 未注解类型的变量

---

## 5. 序列化与跨语言部署

TorchScript 最重要的价值是**脱离 Python 环境部署**：

```python
# 保存
scripted_model.save("model.pt")

# Python 加载
loaded = torch.jit.load("model.pt")

# C++ 加载（无需 Python）
# torch::jit::script::Module module = torch::jit::load("model.pt");
```

这使得模型可以直接嵌入到 C++ 服务、iOS/Android App 或嵌入式设备中运行。

---

## 6. TorchScript 在 LLM 推理链路中的位置

```
训练阶段 (Python)
    │  nn.Module (动态图，易调试)
    │
    ▼ torch.jit.script / trace
    │
TorchScript IR
    │
    ├──→ torch.jit.load (C++ / Java / Mobile)
    │
    ├──→ torch.compile (PyTorch 2.0，进一步优化)
    │
    └──→ TensorRT / ONNX (通过 TorchScript 导出)
```

在实际 LLM 生产部署中（如 vLLM、TorchServe），通常将模型核心 forward 逻辑用 `torch.jit.script` 编译，再交给推理引擎调用，以消除 Python GIL 造成的并发瓶颈。


## 7.TorchScript与Torch FX 对比

一句话总结：TorchScript 是为了“脱离 Python 运行”，而 Torch FX 是为了“用 Python 优化 Python”。

| **特性**          | **TorchScript**                    | **Torch FX**                                   |
| ----------------- | ---------------------------------- | ---------------------------------------------- |
| **主要用途**      | **部署**(在 C++ 服务器上跑)        | **变换**(量化、算子融合、架构分析)             |
| **灵活性**        | 较低（严苛的语法检查）             | 极高（只要是标准的 Tensor 操作都能捕获）       |
| **中间表示 (IR)** | 基于栈的序列化格式 (`.pt`)         | 基于图的 Python 节点对象 (`Graph`)             |
| **动态控制流**    | 支持`if`和`loop`(通过 Script 模式) | 默认不支持（Tracing 会将其“展开”为静态路径） |
| **可读性**        | 差（类似汇编）                     | 强（生成的代码依然是易读的 Python）            |

