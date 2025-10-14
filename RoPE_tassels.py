import torch    
import torch.nn as nn

class RoPE(nn.Module):
    """
    一个简洁的、适合面试手写的旋转位置编码 (RoPE) 实现
    * 核心思想：将位置信息编码为向量的旋转操作
    * 适用于多头注意力机制中的查询（Q）和键（K）矩阵
    * shape说明:
      - 输入: (batch_size * num_heads, seq_len, head_dim)
      - 输出: (batch_size * num_heads, seq_len, head_dim)

    """
    def __init__(self, dim, base=10000):
        """
        初始化RoPE参数。

        Args:
            dim: 每个注意力头的维度（必须是偶数）
            base: 频率计算的基数，控制波长，默认为10000
        """
        super().__init__()  # 关键修复：必须调用父类的初始化方法
        self.dim = dim      # 每个头的维度
        self.base = base    # 频率基数
        self.register_buffer('cache', None)   # 使用register_buffer确保缓存能正确转移到设备上

    def get_cache(self, seq_len):
        """
        预计算旋转位置编码所需的cos和sin值缓存
        
        Args:
            seq_len: 序列长度（token数量）
            
        Returns:
            cache: 形状为 (seq_len, dim//2, 2) 的张量
                   最后一维的2分别对应cos和sin值
        """
        ### 步骤1: 计算频率项 theta_i
        # 我们只对一半的维度计算，因为每个频率项对应一对分量
        # i = [0, 2, 4, ..., dim-2] 对应每个分组的索引
        i = torch.arange(start=0, end=self.dim, step=2, dtype=torch.float32)

        # theta_i = 1/(base^(2i/dim)) 
        # 这是RoPE的核心频率计算公式，控制不同维度的旋转速度
        theta = 1.0 / (self.base ** (i / self.dim))  # 计算频率

        ### 步骤2: 生成位置序列 [0, 1, 2, ..., seq_len-1]
        # 每个位置m都会有不同的旋转角度
        pos = torch.arange(seq_len, dtype=torch.float32)  # 位置索引

        ### 步骤3: 计算每个位置m和每个维度i的旋转角度 m * theta_i
        # torch.outer计算外积：结果矩阵中第m行第i列 = pos[m] * theta[i]
        angles = torch.outer(pos, theta)  # 计算位置与频率的外积

        ### 步骤4: 计算每个角度的cos和sin值，并拼接成缓存
        # torch.stack将cos和sin沿着新维度拼接，将cache形成(seq_len, dim//2, 2)的形状
        cache = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # 计算cos和sin值

        return cache  # cache形状: (seq_len, dim//2, 2)

    def forward(self, x):
        """
        对输入张量应用旋转位置编码
        
        Args:
            x: 输入张量，形状为 (batch_size * num_heads, seq_len, head_dim)
               这是多头注意力中重塑后的Q或K矩阵
               
        Returns:
            x_out: 应用旋转位置编码后的张量，形状与输入相同
        """
        # 获取输入的形状信息
        bs_seq, seq_len, dim = x.shape

        ### 步骤1: 检查并获取缓存
        # 如果缓存不存在或序列长度超过当前缓存，重新计算缓存
        if self.cache is None or self.cache.shape[0] < seq_len:
            self.cache = self.get_cache(seq_len)
            print(self.cache.shape)  # 调试信息，显示缓存形状
            
        ### 步骤2: 重塑输入张量，将最后维度分成多个2维分组
        # 原始形状: (batch_size * num_heads, seq_len, head_dim)
        # 重塑后: (batch_size * num_heads, seq_len, head_dim//2, 2)
        # 这样每个2维向量可以独立进行旋转操作
        x_reshaped = x.reshape(-1, seq_len, dim//2, 2)  # x_reshaped形状: (batch_size * num_heads, seq_len, dim//2, 2)

        ### 步骤3: 扩展缓存维度以匹配输入张量
        # 缓存cache原始形状: (seq_len, dim//2, 2)
        # 扩展后: (1, seq_len, dim//2, 2) 这样可以广播到所有batch和头
        cache = self.cache.unsqueeze(0)  # cache形状: (1, seq_len, dim//2, 2)

        ### 步骤4: 应用旋转操作（核心计算）
        # 将每个2维分组拆分为两个分量
        x0 = x_reshaped[..., 0]  # 每个2维分组的第一个分量 
        x1 = x_reshaped[..., 1]  # 每个2维分组的第二个分量 
        
        # 从缓存中获取对应位置的cos和sin值
        cos = cache[..., 0]  # cos(m * theta_i)
        sin = cache[..., 1]  # sin(m * theta_i)

        # 应用旋转矩阵公式：
        # [x0']   [cos(mθ)  -sin(mθ)]   [x0]
        # [x1'] = [sin(mθ)   cos(mθ)] * [x1]
        out0 = x0 * cos - x1 * sin  # 旋转后的第一个分量
        out1 = x0 * sin + x1 * cos  # 旋转后的第二个分量

        ### 步骤5：恢复原始形状
        # 先将两个分量堆叠回 (..., h//2, 2) 形状
        x_out = torch.stack([out0, out1], dim=-1)
        # 然后将最后两个维度展平，恢复为原始形状 (..., h)
        x_out = x_out.flatten(2)
        
        return x_out
    
# 使用示例和测试代码
def example_usage():
    """
    RoPE使用示例，展示如何在实际中应用
    """
    print("=== RoPE 使用示例 ===")
    
    # 创建RoPE实例
    head_dim = 64  # 每个注意力头的维度
    rope = RoPE(dim=head_dim)
    
    # 模拟输入数据
    # 假设有4个注意力头，序列长度为8
    batch_size_num_heads = 4
    seq_len = 8
    x = torch.randn(batch_size_num_heads, seq_len, head_dim)
    
    print(f"输入形状: {x.shape}")
    print("输入数据示例（第一个token的前6个维度）:")
    print(x[0, 0, :6])
    
    # 应用旋转位置编码
    x_rotated = rope(x)
    
    print(f"\n输出形状: {x_rotated.shape}")
    print("旋转后数据示例（第一个token的前6个维度）:")
    print(x_rotated[0, 0, :6])
    
    # 检查是否有变化
    diff = torch.abs(x - x_rotated).max()
    print(f"\n输入输出最大差异: {diff.item()}")
    if diff > 1e-6:
        print("✓ RoPE成功应用了旋转操作!")
    else:
        print("⚠ 警告: 输入输出完全相同，可能存在问题")
    
    # 验证旋转性质
    print("\n=== 旋转性质验证 ===")
    
    # 创建一个简单的测试向量 [1, 0]
    test_vector = torch.tensor([1.0, 0.0]).reshape(1, 1, 2)
    print(f"测试向量: {test_vector.flatten()}")
    
    # 创建一个特殊的RoPE来测试90度旋转
    test_rope = RoPE(dim=2)
    
    # 修复：创建角度张量而不是直接使用浮点数
    angle_90 = torch.tensor(torch.pi / 2)  # 90度，包装成张量
    # 手动创建缓存为90度旋转
    angles = torch.tensor([[angle_90]])  # 形状: (1, 1)
    test_rope.cache = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    
    # 应用90度旋转
    rotated = test_rope(test_vector)
    print(f"旋转90度后: {rotated.flatten()}")
    print(f"期望结果: [0.0, 1.0]")
    
    # 检查旋转是否正确
    expected = torch.tensor([0.0, 1.0])
    error = torch.abs(rotated.flatten() - expected).sum()
    print(f"旋转误差: {error.item():.6f}")
    if error < 1e-6:
        print("✓ 旋转操作验证成功!")
    else:
        print("⚠ 旋转操作可能有问题")


if __name__ == "__main__":
    example_usage()