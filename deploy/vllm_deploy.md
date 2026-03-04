# vLLM 部署

vLLM 部署 Hugging Face `transformers` 格式模型的过程，本质上是将**动态的 Python 定义**转化为**高效的 C++/CUDA 运行时**。

它并没有直接运行 `transformers` 的原生代码，而是“提取其权重，重写其逻辑”。以下是详细的五个步骤：

---

## 一、 vLLM：大模型部署的“动静结合”艺术

在深度学习领域，动态图（Dynamic Graph） 胜在灵活性，而 静态图（Static Graph） 赢在高性能。vLLM 能够成为目前大语言模型（LLM）部署的行业标配，核心在于其精妙的混合架构设计。

### 1. 宏观层面：动态调度（大脑）

vLLM 在请求处理和逻辑组织上保持了动态性。

* **原因**： LLM 的输入输出长度完全不可预测。
* **表现**： 它利用 Python 的异步调度机制，实时决定每一轮（Iteration）处理哪些请求、如何分配显存块。这种“走一步看一步”的灵活性，是实现 PagedAttention（分页注意力）和 Continuous Batching（连续批处理）的前提，解决了显存碎片化问题，将吞吐量提升了 10 倍以上。

### 2. 微观层面：静态执行（肌肉）

在最底层的计算核心，vLLM 追求的是极致静态化。

* **预编译内核**： vLLM 抛弃了原生的 PyTorch 算子，替换为预先用 C++/CUDA 编写并编译好的静态算子库。
* **算子融合**： 它将矩阵乘法、激活函数和归一化等多个步骤“焊死”在一起。由于这些计算逻辑在模型发布时就已固定，静态化的融合算子能极大减少显存读写，压榨硬件极限。

### 3. 架构优势总结

vLLM 的混合架构实现了**“主厨动态下单，流水线静态出货”**：

| 特性 | vLLM 方案 | 带来的好处 |
| :--- | :--- | :--- |
| **部署门槛** | 动态加载 HF 格式 | 秒级启动，无需像纯静态图那样编译数小时。 |
| **内存利用** | 动态分页管理 | 零浪费，支持数百个并发请求同时在线。 |
| **计算效率** | 静态 CUDA 内核 | 高性能，消除 Python 延迟，对齐 C++ 原生速度。 |

**结论**：vLLM 并没有将模型强行转换为一张死板的“静态图”，而是用一套标准化的静态高性能引擎，去驱动极度灵活的动态推理逻辑。这种“动静结合”的智慧，使其在保证开发效率的同时，成为了生产环境中的吞吐量之王。

---

## 二、 部署流程详解

### 1. 模型解析与权重加载 (Loading & Sharding)

当你指向一个 Hugging Face 路径时，vLLM 首先充当一个“翻译官”：

* **读取 `config.json`：** 获取模型架构（如 `LlamaForCausalLM`）。vLLM 内部有一套自己的模型实现代码，这些代码与 `transformers` 库中的数学逻辑一致，但针对推理进行了极致优化。
* **加载 `safetensors`：** 将权重读入内存。如果指定了多显卡（Tensor Parallelism），vLLM 会在加载时自动进行**张量并行**切分，将矩阵分散到不同 GPU 上。

---

### 2. 算子替换 (Kernel Replacement)

这是 vLLM 快的核心原因。它会跳过 `transformers` 库中通用的 PyTorch 算子，替换为自己定制的 **High-performance Kernels**：

* **PagedAttention 替换：** 将标准的多头注意力（Multi-Head Attention）替换为 PagedAttention 内核。
* **融合算子 (Fused Kernels)：** 将 RMSNorm、激活函数（如 Silu）和残差相加等多个步骤合并为一个 CUDA Kernel，减少显存读写。

---

### 3. KV Cache 预分配 (Memory Profiling)

在启动瞬间，vLLM 会进行一次“显存压力测试”：

* **探测剩余显存：** 计算模型权重占用后，剩下的显存全部划分为固定大小的 **Blocks**（通常为 16 个 Token 一个块）。
* **接管管理权：** 即使模型还没开始预测，vLLM 已经把 90% 以上的显存“占为己有”，作为 KV Cache 内存池。

---

### 4. 连续批处理调度 (Continuous Batching)

传统的 `transformers` 部署（如简单用 Flask 封装）需要等一个 Batch 全跑完才能接下一个。vLLM 采用动态调度：

* **迭代级调度 (Iteration-level Scheduling)：** 每一轮生成一个 Token 后，调度器都会检查是否有新请求进来。
* **即插即用：** 如果 Batch A 还在生成，Batch B 进来了，vLLM 会利用空余的计算资源把 B 直接插进当前的计算循环。

---

### 5. 标准接口封装 (API Serving)

最后，vLLM 启动一个异步的 **FastAPI** 服务器：

* **兼容性：** 提供与 OpenAI 完全一致的 API 路由（如 `/v1/chat/completions`）。
* **流式输出 (Streaming)：** 利用 Python 的 `asyncio`，在底层 C++ 每生成一个 Token 时，立即通过 HTTP Server 推送给用户，实现“打字机”效果。

---

### 总结：vLLM vs 原生 Transformers 推理


| 维度         | Transformers (原生)           | vLLM 部署                          |
| ------------ | ----------------------------- | ---------------------------------- |
| **内存管理** | 静态分配，容易 OOM (显存溢出) | **PagedAttention 动态管理**        |
| **批处理**   | 静态 Batch (等待最慢的请求)   | **Continuous Batching (连续插入)** |
| **底层实现** | 通用 PyTorch 算子             | **定制化 CUDA 融合算子**           |
| **吞吐量**   | 低                            | **极高 (通常提升 10-20 倍)**       |

### 快速实操命令

如果你有一个 Hugging Face 格式的模型目录 `./my_model`，部署只需一行命令：

```bash
# 自动识别 HF 格式，开启 2 张显卡并行，使用 FP8 量化
python -m vllm.entrypoints.openai.api_server \
    --model ./my_model \
    --tensor-parallel-size 2 \
    --quantization fp8
```
