"""
TorchScript 使用方法完整示例
============================

本文件演示 TorchScript 的三种核心使用方式：

1. @torch.jit.script 装饰器 — 脚本化独立函数
2. torch.jit.script()    — 脚本化 nn.Module（保留控制流）
3. torch.jit.trace()     — 追踪 nn.Module（无控制流场景）

并展示：
- 查看生成的 TorchScript IR
- 模型的序列化（save）与反序列化（load）
- script 与 trace 在控制流上的关键差异
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 方式 1：@torch.jit.script 装饰器 — 脚本化独立函数
# ============================================================

@torch.jit.script
def fused_gelu_add(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    一个被 TorchScript 编译的融合函数：先加偏置，再做 GELU 激活。
    装饰器会在函数定义时立即触发 AST 解析和编译。
    注意：必须有明确的类型注解。
    """
    return F.gelu(x + bias)


def demo_script_function():
    print("=" * 60)
    print("方式 1: @torch.jit.script 装饰器（脚本化独立函数）")
    print("=" * 60)

    x    = torch.randn(4, 8)
    bias = torch.zeros(8)

    # 直接调用，与普通函数无异
    output = fused_gelu_add(x, bias)
    print(f"  输入形状: {x.shape}, 输出形状: {output.shape}")

    # 查看 TorchScript IR
    print("\n  生成的 TorchScript IR：")
    print(fused_gelu_add.graph)


# ============================================================
# 方式 2：torch.jit.script() — 脚本化含控制流的 nn.Module
# ============================================================

class BranchModel(nn.Module):
    """
    含 if/else 控制流的模型：根据 flag 参数选择不同的计算路径。
    这类模型只能用 torch.jit.script 处理，torch.jit.trace 会丢失分支。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.fc_pos = nn.Linear(dim, dim)
        self.fc_neg = nn.Linear(dim, dim)
        self.norm   = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, flag: int) -> torch.Tensor:
        # 控制流：根据 flag 决定使用哪条路径
        if flag > 0:
            out = self.fc_pos(x)
        else:
            out = self.fc_neg(x)

        out = self.norm(out)
        out = torch.relu(out)
        return out


def demo_script_module():
    print("\n" + "=" * 60)
    print("方式 2: torch.jit.script()（脚本化 nn.Module，保留控制流）")
    print("=" * 60)

    model = BranchModel(dim=16).eval()
    x     = torch.randn(2, 16)

    # --- 2a. 脚本化 ---
    scripted = torch.jit.script(model)
    print("  脚本化成功 ✅")

    # --- 2b. 验证两条分支都能正确执行 ---
    with torch.no_grad():
        out_pos = scripted(x, flag=1)   # 走 fc_pos 路径
        out_neg = scripted(x, flag=-1)  # 走 fc_neg 路径
    print(f"  flag=+1 输出: {out_pos.shape}, flag=-1 输出: {out_neg.shape}")
    print(f"  两条路径输出是否相同（预期为 False）: {torch.allclose(out_pos, out_neg)}")

    # --- 2c. 查看包含 prim::If 的 IR ---
    print("\n  生成的 TorchScript IR（含 prim::If 控制流节点）：")
    print(scripted.graph)

    # --- 2d. 序列化与加载 ---
    save_path = "computational graph/ir/branch_model_scripted.pt"
    scripted.save(save_path)
    loaded = torch.jit.load(save_path)
    with torch.no_grad():
        out_loaded = loaded(x, flag=1)
    is_same = torch.allclose(out_pos, out_loaded, atol=1e-6)
    print(f"\n  序列化后重新加载，结果一致: {is_same} ✅")


# ============================================================
# 方式 3：torch.jit.trace() — 追踪 nn.Module（无控制流场景）
# ============================================================

class SimpleEncoder(nn.Module):
    """
    一个纯前馈编码器，无数据依赖的控制流，适合用 trace 处理。
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc1  = nn.Linear(in_dim, out_dim * 2)
        self.fc2  = nn.Linear(out_dim * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.norm(x)
        return x


def demo_trace_module():
    print("\n" + "=" * 60)
    print("方式 3: torch.jit.trace()（追踪 nn.Module，无控制流场景）")
    print("=" * 60)

    model         = SimpleEncoder(in_dim=32, out_dim=16).eval()
    example_input = torch.randn(4, 32)  # 必须提供示例输入

    # --- 3a. 追踪 ---
    traced = torch.jit.trace(model, example_input)
    print("  追踪成功 ✅")

    # --- 3b. 验证输出 ---
    with torch.no_grad():
        out_original = model(example_input)
        out_traced   = traced(example_input)
    is_same = torch.allclose(out_original, out_traced, atol=1e-6)
    print(f"  原始模型与 traced 模型输出一致: {is_same} ✅")

    # --- 3c. 查看 IR ---
    print("\n  生成的 TorchScript IR（纯线性算子序列，无控制流）：")
    print(traced.graph)

    # --- 3d. 序列化 ---
    save_path = "computational graph/ir/simple_encoder_traced.pt"
    traced.save(save_path)
    loaded = torch.jit.load(save_path)
    with torch.no_grad():
        out_loaded = loaded(example_input)
    print(f"\n  序列化后重新加载，结果一致: {torch.allclose(out_original, out_loaded, atol=1e-6)} ✅")


# ============================================================
# 关键差异演示：trace 会"丢失"控制流
# ============================================================

def demo_trace_loses_control_flow():
    print("\n" + "=" * 60)
    print("⚠️  关键差异：trace 会丢失控制流（重要！）")
    print("=" * 60)

    model = BranchModel(dim=16).eval()
    x     = torch.randn(2, 16)

    # trace 时传入 flag=1，只会录制 fc_pos 那条路径
    print("  用 flag=1 追踪 BranchModel（只记录 fc_pos 路径）...")
    traced_wrong = torch.jit.trace(model, (x, torch.tensor(1)))

    with torch.no_grad():
        # 即使传入 flag=-1，traced 模型仍然走 fc_pos 路径！
        out_script_neg = torch.jit.script(model)(x, flag=-1)
        out_trace_neg  = traced_wrong(x, torch.tensor(-1))

    is_wrong = not torch.allclose(out_script_neg, out_trace_neg, atol=1e-4)
    print(f"  flag=-1 时，script 与 trace 输出是否不同（预期 True）: {is_wrong}")
    print("  → trace 版本错误地忽略了 flag=-1，始终走 fc_pos 分支！")
    print("  → 结论：包含控制流的模型必须使用 torch.jit.script ！")


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    os.makedirs("computational graph/ir", exist_ok=True)

    demo_script_function()
    demo_script_module()
    demo_trace_module()
    demo_trace_loses_control_flow()

    print("\n" + "=" * 60)
    print("所有演示完成。序列化文件已保存至 computational graph/ir/")
    print("=" * 60)

