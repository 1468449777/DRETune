import numpy as np

# Step 1: 生成一个低维向量 d 和投影矩阵 A
n = 10  # 低维空间的维度
N = 25  # 高维空间的维度

# 随机生成一个低维向量 d（n维）
d = np.random.rand(n)

# 随机生成一个 N x n 的投影矩阵 A（可能是一个非方阵）
A = np.random.rand(N, n)

# Step 2: 计算高维向量 D
D = np.dot(A, d)

# Step 3: 使用伪逆恢复低维向量 d_hat
A_plus = np.linalg.pinv(A)  # 计算 A 的伪逆
d_hat = np.dot(A_plus, D)  # 恢复低维向量 d

# Step 4: 输出结果并验证
print("原始低维向量 d:", d)
print("计算得到的高维向量 D:", D)
print("通过伪逆恢复的低维向量 d_hat:", d_hat)

# 验证恢复的 d_hat 是否接近原始的 d
print("恢复的低维向量 d_hat 与原始 d 的差异:", np.linalg.norm(d_hat - d))



import numpy as np

# 假设 s 是一个 shape 为 (500, 28) 的数组
s = np.random.rand(5, 1)  # 示例数据

# 使用 np.tile 重复 s 100 次，得到 shape 为 (50000, 28) 的数组
n = np.tile(s, (10, 1))  # 沿第一个维度重复 100 次

# 检查结果
print(n.shape)  # 输出: (50000, 28)
print("s = ", s)
print("n = ", n)

