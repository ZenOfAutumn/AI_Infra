import torch
import torch.nn as nn

class StaticGraphModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StaticGraphModel, self).__init__()
        # 初始化权重 W1, W2 和偏置 b
        self.W1 = nn.Parameter(torch.randn(output_dim, input_dim))
        self.W2 = nn.Parameter(torch.randn(output_dim, input_dim))
        self.b = nn.Parameter(torch.randn(output_dim, 1))

    # 在 TorchScript 中，需要显式指定参数的类型注解，以便编译器正确解析控制流
    def forward(self, X: torch.Tensor, flag: int) -> torch.Tensor:
        if flag > 0:
            Y = torch.matmul(self.W1, X)
        else:
            Y = torch.matmul(self.W2, X)

        Y = Y + self.b
        Y = torch.relu(Y)
        return Y

if __name__ == "__main__":
    input_dim = 4
    output_dim = 3

    # 1. 实例化动态图模型
    dynamic_model = StaticGraphModel(input_dim, output_dim)

    # 2. 将动态图转换为静态计算图 (TorchScript)
    # 注意：因为代码中包含 if/else 控制流，必须使用 torch.jit.script 而不能使用 torch.jit.trace。
    # trace 只会记录一次执行的路径（即只保留 if 或只保留 else），而 script 会解析 AST 并保留完整的控制流图。
    static_model = torch.jit.script(dynamic_model)

    # 3. 测试静态图模型
    X = torch.randn(input_dim, 1)

    print("=== 测试 flag > 0 ===")
    out_positive = static_model(X, 1)
    print(out_positive)

    print("\n=== 测试 flag <= 0 ===")
    out_negative = static_model(X, -1)
    print(out_negative)

    # 4. 打印生成的静态计算图 (TorchScript IR)
    print("\n=== 静态计算图的中间表示 (IR) ===")
    print(static_model.graph)

