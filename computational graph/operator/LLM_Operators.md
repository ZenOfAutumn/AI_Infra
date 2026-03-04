# 业界主流 LLM 模型核心 PyTorch 算子解析

大语言模型（LLM）如 GPT、LLaMA、Qwen 等，其底层计算图由一系列基础算子（Operators）构成。本文将业界主流 LLM 模型中高频使用的 PyTorch 算子按照**张量操作算子**、**神经网络算子**、**数据流算子**和**控制流算子**进行分类，并提供具体的使用示例。

---

## 1. 张量操作算子 (Tensor Operations)

张量操作是 LLM 计算的基础，主要涉及形状变换、维度重排、切片以及基础数学运算。

### 1.1 形状变换与重排
在 Multi-Head Attention (MHA) 中，经常需要将张量在 `[batch_size, seq_len, hidden_size]` 和 `[batch_size, num_heads, seq_len, head_dim]` 之间转换。

* **`view` / `reshape`**: 改变张量形状。
* **`transpose` / `permute`**: 交换张量维度。

**示例：MHA 中的 QKV 拆分与重排**
```python
import torch

batch_size, seq_len, num_heads, head_dim = 2, 10, 8, 64
hidden_size = num_heads * head_dim

# 假设 qkv_states 是 Linear 层的输出
qkv_states = torch.randn(batch_size, seq_len, 3 * hidden_size)

# 1. view: 拆分出 num_heads 和 head_dim
qkv_states = qkv_states.view(batch_size, seq_len, 3, num_heads, head_dim)

# 2. permute: 将 num_heads 移到前面，形状变为 [batch, 3, num_heads, seq_len, head_dim]
qkv_states = qkv_states.permute(0, 2, 3, 1, 4)

# 3. chunk: 沿维度 1 拆分为 Q, K, V
q, k, v = qkv_states.chunk(3, dim=1)
# q 形状: [batch, 1, num_heads, seq_len, head_dim]
# squeeze: 去掉大小为 1 的维度
q = q.squeeze(1) # [batch, num_heads, seq_len, head_dim]
```

### 1.2 基础数学运算
* **`matmul` / `@`**: 矩阵乘法，LLM 中计算量最大的算子（用于 Linear 层和 Attention Score 计算）。
* **`mul` / `*`**: 逐元素相乘（如 SwiGLU 激活函数中的门控机制）。

**示例：计算 Attention Score**
```python
# q: [batch, num_heads, seq_len, head_dim]
# k: [batch, num_heads, seq_len, head_dim]
# 需要将 k 的最后两个维度转置才能相乘
attn_weights = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)
# attn_weights 形状: [batch, num_heads, seq_len, seq_len]
```

---

## 2. 神经网络算子 (Neural Network Operators)

这些算子构成了 Transformer 架构的核心组件。

### 2.1 归一化 (Normalization)
LLaMA 等现代模型普遍采用 RMSNorm 替代传统的 LayerNorm，以提高计算效率。

* **`nn.LayerNorm`**: 标准层归一化。
* **`RMSNorm` (自定义或第三方库)**: 均方根归一化。

**示例：RMSNorm 的 PyTorch 实现**
```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 计算均方根
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

x = torch.randn(2, 10, 512)
norm = RMSNorm(512)
out = norm(x)
```

### 2.2 激活函数 (Activations)
现代 LLM 的 MLP 层通常使用 SwiGLU 或 GELU。

* **`F.silu`**: 即 Swish 激活函数，用于 SwiGLU。
* **`F.gelu`**: GPT 系列常用的激活函数。

**示例：LLaMA 中的 SwiGLU 实现**
```python
import torch.nn.functional as F

# 假设 x 经过两个线性层得到 gate 和 up
gate = torch.randn(2, 10, 1024)
up = torch.randn(2, 10, 1024)

# SwiGLU 计算: SiLU(gate) * up
activated = F.silu(gate) * up
```

### 2.3 核心注意力 (Attention)
* **`F.scaled_dot_product_attention` (SDPA)**: PyTorch 2.0 引入的融合算子，底层自动调用 FlashAttention 或 Memory-Efficient Attention，极大降低显存占用。

**示例：使用 SDPA**
```python
q = torch.randn(2, 8, 10, 64)
k = torch.randn(2, 8, 10, 64)
v = torch.randn(2, 8, 10, 64)

# is_causal=True 自动应用下三角掩码 (Causal Mask)
attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

---

## 3. 数据流算子 (Data Flow Operators)

在 LLM 推理（特别是自回归生成阶段），数据流算子用于管理 KV Cache 和处理序列拼接。

### 3.1 拼接与堆叠
* **`cat`**: 沿现有维度拼接张量（常用于 KV Cache 的更新）。
* **`stack`**: 沿新维度堆叠张量。

**示例：更新 KV Cache**
```python
# 历史的 K cache: [batch, num_heads, past_seq_len, head_dim]
past_key_cache = torch.randn(2, 8, 5, 64)
# 当前 step 新生成的 K: [batch, num_heads, 1, head_dim]
new_key = torch.randn(2, 8, 1, 64)

# 将新生成的 K 拼接到历史 Cache 中
updated_key_cache = torch.cat([past_key_cache, new_key], dim=2)
# 形状变为: [batch, 8, 6, 64]
```

### 3.2 索引与收集
* **`gather` / `take_along_dim`**: 根据索引提取元素（常用于从 logits 中提取特定 token 的概率）。

**示例：提取目标 Token 的 Logits**
```python
logits = torch.randn(2, 10, 32000) # [batch, seq_len, vocab_size]
# 获取最后一个 token 的 logits 用于预测下一个词
next_token_logits = logits[:, -1, :] # [batch, 32000]
```

---

## 4. 控制流算子 (Control Flow Operators)

控制流算子用于处理掩码（Masking）、条件选择以及位置编码的生成。

### 4.1 掩码操作
* **`masked_fill`**: 根据布尔掩码将特定位置填充为指定值（常用于 Attention Mask，将 padding 或未来 token 的 score 设为负无穷）。

**示例：生成 Causal Mask**
```python
seq_len = 5
# 生成上三角矩阵 (1 表示需要 mask 的位置)
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
# mask:
# [[False,  True,  True,  True,  True],
#  [False, False,  True,  True,  True],
#  ...]

attn_scores = torch.randn(seq_len, seq_len)
# 将需要 mask 的位置填充为 -inf
attn_scores = attn_scores.masked_fill(mask, float('-inf'))
```

### 4.2 条件选择
* **`where`**: 根据条件从两个张量中选择元素。

**示例：混合精度下的 Mask 处理**
```python
condition = mask == True
# 如果 condition 为 True，选 -1e4，否则保留原值
safe_scores = torch.where(condition, torch.tensor(-1e4), attn_scores)
```

### 4.3 序列生成
* **`arange` / `cumsum`**: 生成序列或累加（常用于生成位置 ID，进而计算 RoPE 旋转位置编码）。

**示例：生成 Position IDs**
```python
attention_mask = torch.tensor([[1, 1, 1, 0, 0], [0, 1, 1, 1, 1]])
# 利用 cumsum 计算实际的 position_ids
position_ids = (torch.cumsum(attention_mask, dim=1) - 1).clamp(min=0)
# position_ids:
# [[0, 1, 2, 2, 2],
#  [0, 0, 1, 2, 3]]
```

---

## 总结

LLM 的前向传播本质上是上述算子的有向无环图（DAG）。在实际的工业级部署中（如 vLLM, TensorRT-LLM），为了追求极致性能，通常会将多个细粒度的 PyTorch 算子**融合（Kernel Fusion）**成一个自定义的 CUDA/Triton 算子。例如：
* `matmul` + `add` + `silu` + `mul` 融合为 **Fused SwiGLU**。
* `view` + `permute` + `matmul` + `masked_fill` + `softmax` + `matmul` 融合为 **FlashAttention**。
* `cat` (KV Cache 更新) 演变为底层的 **PagedAttention** 显存指针操作。

