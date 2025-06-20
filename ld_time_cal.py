# 测试双随机投影降维所需的时间

from hes.low_dim_adaptor import LowDimAdaptor
import numpy as np
import time


HES_LOW_ACT_DIM = HES_LOW_ACT_DIM_DOUBLE = 90
HES_HIGH_ACT_DIM = HES_HIGH_ACT_DIM_DOUBLE = 132

HES_LOW_SEED = 40

NODE_NUM = 12

USE_DOUBLE = False

ld_adaptor = LowDimAdaptor(None, HES_LOW_ACT_DIM, HES_LOW_SEED, HES_HIGH_ACT_DIM)
ld_adaptor_double = LowDimAdaptor(None, HES_LOW_ACT_DIM_DOUBLE, HES_LOW_SEED, HES_HIGH_ACT_DIM_DOUBLE)
ld_adaptor_double.double_network_correct(ld_adaptor.A)
ld_adaptor_double.double_network_correct(ld_adaptor.A)




CAL_TIME = 6011100

print(ld_adaptor.A.shape)



total_time = 0
mul_time = 0
for i in range(CAL_TIME):
    # 生成形状为 (HES_LOW_ACT_DIM, NODE_NUM) 的矩阵
    matrix = np.random.uniform(-1, 1, (HES_LOW_ACT_DIM, NODE_NUM))
    if USE_DOUBLE:
        start_time = time.time()
        hd1 = ld_adaptor.transform(matrix)
        hd2 = ld_adaptor_double.transform(matrix)
        medium_time = time.time()
        hd = hd1 + hd2
        end_time = time.time()
        total_time += end_time - start_time
        mul_time += medium_time - start_time
        
    else:
        start_time = time.time()
        hd1 = ld_adaptor.transform(matrix)
        medium_time = time.time()
        hd = hd1
        end_time = time.time()
        total_time += end_time - start_time
        mul_time += medium_time - start_time
add_time = total_time - mul_time 
print("total time = {0} s, multiply time = {1} s, add time = {2} s".format(total_time, mul_time, add_time))
        



# ld_adaptor.transform()







