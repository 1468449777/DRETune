# 测试修改参数后性能稳定的时间
from maEnv.env import CEEnv
from maEnv import utils
import time



def test_p():
    env = CEEnv()
    # 每5秒输出一次qps
    for i in range(1000):
        qps = utils.get_qps()
        print('qps = ', qps)
        # time.sleep(5)


def test_mapping(a, min, max):
    return ((max - min) / 2) * a + (max + min) / 2



if __name__ == '__main__':
    a = test_mapping(-0.9,0,10)
    print(a)