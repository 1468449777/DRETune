import paddle.fluid as fluid
import numpy as np

# 启用动态图模式
with fluid.dygraph.guard():
    # 创建一个 Paddle Variable
    input_var = fluid.dygraph.to_variable(np.random.rand(3, 4).astype('float32'))

    # 预设一个矩阵
    matrix = np.random.rand(4, 5).astype('float32')
    matrix_var = fluid.dygraph.to_variable(matrix)

    # 矩阵乘法
    result = fluid.layers.matmul(input_var, matrix_var)

    print("Input Shape:", input_var.shape)
    print("Matrix Shape:", matrix_var.shape)
    print("Result Shape:", result.shape)
