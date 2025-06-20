# 训练入口
# 开启训练过程指令：python main.py --algorithm SAC_2
# 挂后台跑：nohup python -u main.py --algorithm SAC_2 > {log_dir_path}/log_{name}_{date}.log &
import threading
import argparse
from tune import train as train
from maEnv import globalValue


def train_algorithm(algorithm):
    matuner = train.MATuner("RL", "SAC_2")
    matuner.train(mode=1)


if __name__ == '__main__':
    # 1.解析命令行参数
    parser = argparse.ArgumentParser(description="训练入口")
    parser.add_argument('--mode', type=str, required=True, help="算法名称，例如 SAC_2")
    args = parser.parse_args()

    # 2.先进行一些全局变量的初始化工作
    globalValue.LOAD_EVENT = threading.Event()

    # 3.之后开启训练线程
    # 训练线程开启
    print(globalValue.BASE_HOME)
    train_thread = threading.Thread(target=train_algorithm, args=(args.mode,))
    train_thread.start()