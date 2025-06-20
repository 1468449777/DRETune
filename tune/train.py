#import gym
import csv
import os
import math
from math import sqrt

import gym
import pandas as pd
import numpy as np
import parl
from parl.utils import logger, action_mapping
import scikitplot as skplt
# from factor_analyzer.factor_analyzer import calculate_kmo
#from rlschool import make_env
from parl import layers
import random

from torch import Tensor

from maEnv.utils import get_timestamp, time_to_str
from my_algorithm.agent import Agent
from my_algorithm.model import Model
# from my_algorithm.algorithm import DDPG  # from parl.algorithms import DDPG
from parl.algorithms import DDPG  # from parl.algorithms import TD3
#from parl.algorithms import TD3  # from parl.algorithms import TD3
from my_algorithm.td3_ import TD3
from my_algorithm.sac import SAC
from my_algorithm.sac_2 import SAC_2
# from parl.algorithms import DQN
# from my_algorithm.algorithm import TD3  # from parl.algorithms import TD3
# from my_algorithm.TD3 import TD3

# 用于降维的库
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from my_algorithm.td3_model import TestModel
from my_algorithm.sac_model import SACModel
from my_algorithm.agent import TestAgent
from my_algorithm.agent import SACAgent
from my_algorithm.agent import SAC2Agent
from my_algorithm.DQN_model import Model as DQNModel
from my_algorithm.DQN_model import Agent as DQNAgent
from my_algorithm.DQN_model import DQN

from my_algorithm.replay_memory import ReplayMemory
from my_algorithm.priority_replay_memory import PrioritizedReplayMemory
import pickle
import time
from maEnv import globalValue
from maEnv.env import SEEnv
from maEnv.env import CEEnv
from maEnv.env import NodesEnv
from maEnv import utils
from maEnv import datautils
from maEnv import utils

from my_algorithm.LERC import LERC
from hes.low_dim_adaptor import LowDimAdaptor
from model_transfer import make_model

import multiprocessing


# h_params of TD3:
LEARNING_RATE = 0.001
ACTOR_LR = 0.0001  # Actor网络的 learning rate
CRITIC_LR = 0.0002 # Critic网络的 learning rate
GAMMA = 0.95  # reward 的衰减因子
TAU = 0.005  # 软更新的系数


# h_params of SAC:
H_SAC_ACTOR_LR = 0.0001     # Actor网络的 learning rate
H_SAC_CRITIC_LR = 0.0002    # Critic网络的 learning rate
H_SAC_GAMMA = 0.95          # reward 的衰减因子
H_SAC_TAU = 0.005           # 软更新的系数
H_SAC_ALPHA = 0.2          # 温度参数，决定了熵相对于奖励的相对重要性
H_SAC_ALPHA_LR = 0.0001



MEMORY_SIZE = 100000  # 经验池大小
MEMORY_WARMUP_SIZE = 0 #120  # 预存一部分经验之后再开始训练
BATCH_SIZE = 2
WARMUP_MOVE_STEPS = 1
MOVE_STEPS = 1 #20
REWARD_SCALE = 1  # reward 缩放系数
# TRAIN_EPISODE = 100  # 训练的总episode数
TRAIN_EPISODE = 30 #15  # 训练的总episode数
EVAL_INTERVAL = 5   # 评估的间隔
EXPL_NOISE = 0.2   # 动作噪声方差
EXPL_NOISE_WARMUP = 0.3
POLICY_NOISE = 0.1  # Noise added to target policy during critic update
NOISE_CLIP = 0.5  # Range to clip target policy noise
POLICY_FREQ = 2
# ENV_METHOD = 'TD3'
ENV_METHOD = 'SAC'  # 选择的DRL方法(废弃)
ACTION_TREND_P = 0.5#0.995  # 专家知识的选择概率
BEST_NOW_P = 0.005
RANDOM_FOREST_REMAIN = 1
PCA_REMAIN = 10
TWO_PHASE = False


# 参数降维
USE_KNOBS_DR = False
# 状态向量降维
USE_STATUS_DR = False
# 使用专家经验开关
USE_EXPERT_EXP = False


# 降维时超拉丁方采样开关（基于排名的降维方法,废弃）
USE_LHS = False


# 经验池优先经验回放开关(废弃)
USE_PRIORITY_RPM = False           

# 随机森林判断阈值
RF_THRESHOLD = 0.7

# 使用固定降维参数开关(废弃,也许还能用)
USE_FIXED_DR_KNOBS = False

# 使用专家生成经验快速学习的开关
USE_EXP_GEN_FAST_LERAN = False
# label/action ratio(废弃)
EXP_GEN_S_RATIO = [0.4, 0.3, 0.3]
EXP_GEN_R_RATIO = [0.5, 0.5]
# 专家生成经验的总量
EXP_GEN_TOTAL_NUM = 500000
# 专家生成经验快速学习的轮次
EXP_GEN_LEARN_EPISODE = 10000
# 专家生成经验快速学习batchsize
EXP_GEN_LEARN_BATCH_SIZE = 2000
# 使用真实环境数据测试，真实数据文件路径(废弃)
EXP_GEN_ACTUAL_TEST_FILE = "/home/orange/MATune/matune_wxc/wxc_test/first_phase_2024-06-17_00:24:05_nodes_rpm.txt"
# 使用真实环境数据测试，真实环境数据测试时采样的数量(废弃)
EXP_GEN_ACTUAL_ENV_SAMPLE_NUM = 79
# 使用生成数据测试，早停的命中率阈值(废弃)
EXP_GEN_GEN_TEST_RATE_THRESHOLD = 0.75
# 使用真实环境数据测试，早停的命中率阈值(废弃)
EXP_GEN_ACTUAL_ENV_RATE_THRESHOLD = 0.7
# 使用专家生成经验快速学习的阶段(不用改这个参数)
EXP_GEN_FAST_LERAN_STAGE = [0]
# 快速学习训练停止的标准阈值
EXP_GEN_HIT_THRES = globalValue.FAST_LEARN_HIT_THRES 
# 使用真实数据测试开关(废弃,一直设置为关即可)
EXP_GEN_USE_ACTUAL_ENV_TEST = False





# 手动限定使用的经验池(废弃)
USE_FIXED_RPM = False
FIXED_RPM_PATH = "./test_wxc/fixed_rpm_rw_l.txt"
FIXED_AR_PATH = "./test_wxc/fixed_ar_rw_l.csv"


# 模拟退火算法相关参数（废弃）
SA_INIT_TEMPERTURE = 100
SA_ALPHA = 0.95
SA_MAX_ITER = 10
SA_BASH_SLEEP_TIME = 20
SA_BASH_BUFFER_TIME = 10    
SA_BASH_TYPE = 'sysbench'
SA_NEW_RANDOM_ACTION_P = 0.5
SA_OLD_ACTION_BIAS = 0.5

# 低维随机投影开关
USE_LD_ADAPTOR = True           # 是否开启低位投影
USE_DOUBLE_NETWORK = True       # 是否开启双网络
USE_GRID_SEARCH = False         # 是否开启 grid_serach,开启之后双网络的效果更好,强烈建议打开

USE_HIS_MODEL = False           # 废弃
USE_HIS_RPM = False              # 迁移开关
HIS_TRANSFER_RPM_PATH = f"{globalValue.BASE_HOME}/DRLETune/transfer_rpm/rw_backup/rpm_episode_18.txt"    # 历史模型产生的经验池
HIS_TRANSFER_RPM_DOUBLE_PATH = f"{globalValue.BASE_HOME}/DRLETune/transfer_rpm/rw_backup/rpm_double_episode_18.txt" # 历史模型产生的经验池(用于double网络)
HIS_TRANSFER_BATCH = 2000   # 迁移时学习的batch_size
HIS_TRANSFER_ITER = 1000    # 迁移时学习的轮次
HIS_TRANSFER_PRINT_ITER = 10    # 迁移时打印的间隔



# 改造
# 目标：将原来以函数为主体的训练方法（面向过程）改造成以模型的类为主体（面向对象）
class MATuner:
    
    # 初始化函数
    def __init__(self, model_type, model_name):
        self.model_type = model_type            # model的类型：目前支持两种，强化学习方法（rl）/启发时算法（ha）
        self.model_name = model_name            # model的具体名称：比如TD3、SA
        self.model = None
        self.agent = None
        self.env = None
        self.rpm = None
        self.pca = None
        self.TD3_logger = None
        # double network
        if USE_DOUBLE_NETWORK:
            self.model_double=None
            self.agent_double = None
            self.rpm_double = None
        
        self.rpm_full_real = None
        
    # 主要的训练函数, 会调用run_episode 
    def train(self, mode = 1):

        print("------MultiNodes Train (use {0} {1}) thread start...------".format(self.model_type, self.model_name))
        if self.model_type == "RL":
            pass
        
        self.env = env = NodesEnv()
        env.init_nodes()
        if env.method == 'DQN':
            env.init_env_for_DQN()
            
        # 初始化双随即投影的种子
        self.hes_low_seed = hes_low_seed = globalValue.HES_LOW_SEED
        self.hes_low_seed_double = hes_low_seed_double = globalValue.HES_LOW_SEED
            
        print("env.action_dim = ", env.action_dim)
        
        # 双随即投影矩阵的生成和检查
        if USE_LD_ADAPTOR:
            env.ld_adaptor = LowDimAdaptor(env, globalValue.HES_LOW_ACT_DIM, globalValue.HES_LOW_SEED)
            if USE_DOUBLE_NETWORK:
                env.ld_adaptor_double = LowDimAdaptor(env, globalValue.HES_LOW_ACT_DIM_DOUBLE, hes_low_seed+globalValue.DOUBLE_SEED_DELTA)
                # TODO 对生成做判断(具体可以看一下论文,基本是按照论文里所讲的做的)
                env.ld_adaptor_double.double_network_correct(env.ld_adaptor.A)
                print("===============================================================")
                env.ld_adaptor_double.double_network_correct(env.ld_adaptor.A)
            env.action_dim = env.ld_adaptor.target_dim
            
        print("env.action_dim = ", env.action_dim)

        
        env.method = self.model_name
        env.expert_exp = USE_EXPERT_EXP
        env.use_exp_fl = USE_EXP_GEN_FAST_LERAN or USE_HIS_RPM
        obs_dim = env.state_dim
        act_dim = env.action_dim
        max_action = 0.99999999

        if env.expert_exp == True:
            env.action_trend_choice = ACTION_TREND_P
        else:
            env.action_trend_choice = 0
        env.best_action_choice = 0
        
        
        if USE_LHS:
            env.lhs_sampler = utils.LHSampler(WARMUP_MOVE_STEPS, env.action_dim)
        
        if USE_HIS_RPM:
            USE_EXP_FAST_GEN_LERAN= False
            env.lerc = LERC(obs_dim, act_dim, True, BATCH_SIZE, len(globalValue.CONNECT_CE_IP) + len(globalValue.CONNECT_SE_IP), USE_LD_ADAPTOR, env.ld_adaptor)

        
        
        f_fl_train_loss = open("./test_model/fl_train_loss.log", 'w+', encoding="utf-8")
        f_fl_train_loss.close()
        f_fl_time = open("./test_model/fl_time.log", 'w+', encoding="utf-8")
        f_fl_time.close()
        fl_start_time = time.time()
        
        # 开启基于专家知识的快速趋势学习
        if USE_EXP_GEN_FAST_LERAN:
            
            # 初始化LERC对象(专门用来生成经验样本的)
            env.lerc = LERC(obs_dim, act_dim, True, BATCH_SIZE, len(globalValue.CONNECT_CE_IP) + len(globalValue.CONNECT_SE_IP), USE_LD_ADAPTOR, env.ld_adaptor)
            # 生成专家模拟经验
            exp_s_list, exp_a_list, exp_r_list, exp_n_s_list, exp_d_list, exp_l_a_list, exp_trend_list = \
                env.lerc.generate_expert_exp(EXP_GEN_TOTAL_NUM, env)
            
            print("GENERTATE {0} EXPERT EXP".format(EXP_GEN_TOTAL_NUM))
            print("exp_s_list shape = ", exp_s_list.shape)
            print("exp_a_list shape = ", exp_a_list.shape)
            print("exp_r_list shape = ", exp_r_list.shape)
            print("exp_n_s_list shape = ", exp_n_s_list.shape)
            print("exp_d_list shape = ", exp_d_list.shape)
            print("exp_l_a_list shape = ", exp_l_a_list.shape)
            print("exp_trend_list shape = ", exp_trend_list.shape)
            
            # rpm_act_dim = act_dim
            # if USE_LD_ADAPTOR
            # 初始化用于趋势学习的双随即投影矩阵和经验池
            ld = act_dim
            hd = act_dim
            if USE_LD_ADAPTOR:
                ld = env.ld_adaptor.target_dim
                hd = env.ld_adaptor.high_dim
            exp_gen_rpm_train = ReplayMemory(MEMORY_SIZE, act_dim, obs_dim, USE_LD_ADAPTOR, ld, hd)
            exp_gen_rpm_test = ReplayMemory(MEMORY_SIZE, act_dim, obs_dim, USE_LD_ADAPTOR, ld, hd)
            # 生成的经验池(实际我们跑的时候发现只用生成的经验即可达到比较好的效果,如果还想用负载下真实的状态生成经验可以参考迁移中的代码)
            for i in range(int(EXP_GEN_TOTAL_NUM * 0.8)):
                exp_gen_rpm_train.append((exp_s_list[i], exp_a_list[i], exp_r_list[i], exp_n_s_list[i], exp_d_list[i], exp_l_a_list[i], exp_trend_list[i]))
            for i in range(int(EXP_GEN_TOTAL_NUM * 0.8), EXP_GEN_TOTAL_NUM):
                exp_gen_rpm_test.append((exp_s_list[i], exp_a_list[i], exp_r_list[i], exp_n_s_list[i], exp_d_list[i], exp_l_a_list[i], exp_trend_list[i]))
                
            # 真实环境的经验池(废弃)
            if EXP_GEN_USE_ACTUAL_ENV_TEST:
                exp_gen_rpm_actual_env_file = open(EXP_GEN_ACTUAL_TEST_FILE, "rb")
                exp_gen_rpm_actual_env = pickle.load(exp_gen_rpm_actual_env_file)
                
        # 一些计时的代码,可以不用太管
        fl_gen_exp_end_time = time.time()
        fl_gen_exp_time = fl_gen_exp_end_time - fl_start_time
        print("[FAST LEARN TIME] gen_exp_time = {:.4f} s".format(fl_gen_exp_time))
        
        with open("./test_model/fl_time.log", "a+", encoding="utf-8") as f_fl_time:
            f_fl_time.write("[FAST LEARN TIME] gen_exp_time = {:.4f} s\n".format(fl_gen_exp_time))
        
        
        
        
        # 一系列初始化模型的代码
        # SAC
        if env.method == 'SAC':
            self.model =  SACModel(act_dim, self.env.state_dim)
            algorithm = SAC(
                actor=self.model.actor_model,
                critic=self.model.critic_model,
                max_action=max_action,
                alpha=H_SAC_ALPHA,
                gamma=H_SAC_GAMMA,
                tau=H_SAC_TAU,
                actor_lr=H_SAC_ACTOR_LR,
                critic_lr=H_SAC_CRITIC_LR,
            )
            self.agent = SACAgent(algorithm, obs_dim, act_dim)
        
        # SAC_2
        if env.method == 'SAC_2':
            self.model =  SACModel(act_dim, self.env.state_dim)
            algorithm = SAC_2(
                actor=self.model.actor_model,
                critic=self.model.critic_model,
                alpha_model=self.model.alpha_model,
                reward_model=self.model.reward_model,
                max_action=max_action,
                alpha=H_SAC_ALPHA,
                gamma=H_SAC_GAMMA,
                tau=H_SAC_TAU,
                actor_lr=H_SAC_ACTOR_LR,
                critic_lr=H_SAC_CRITIC_LR,
                alpha_lr=H_SAC_ALPHA_LR,
                batch_size=BATCH_SIZE,
                states_model=self.model.states_model,
                
            )
            if USE_DOUBLE_NETWORK:
                self.model_double =  SACModel(env.ld_adaptor_double.target_dim, self.env.state_dim, True)
                algorithm_double = SAC_2(
                    actor=self.model_double.actor_model,
                    critic=self.model_double.critic_model,
                    alpha_model=self.model_double.alpha_model,
                    reward_model=self.model_double.reward_model,
                    max_action=max_action,
                    alpha=H_SAC_ALPHA,
                    gamma=H_SAC_GAMMA,
                    tau=H_SAC_TAU,
                    actor_lr=H_SAC_ACTOR_LR,
                    critic_lr=H_SAC_CRITIC_LR,
                    alpha_lr=H_SAC_ALPHA_LR,
                    batch_size=BATCH_SIZE,
                    states_model=self.model_double.states_model,
                    is_double=True,
                )
            ld = act_dim
            hd = act_dim
            if USE_LD_ADAPTOR:
                ld = env.ld_adaptor.target_dim
                hd = env.ld_adaptor.high_dim
            self.agent = SAC2Agent(algorithm, obs_dim, act_dim, ld, hd)
            if USE_DOUBLE_NETWORK:
                self.agent_double = SAC2Agent(algorithm_double, obs_dim, env.ld_adaptor_double.target_dim, env.ld_adaptor_double.target_dim, hd)


            if USE_HIS_MODEL:
                self.agent.restore("./test_model/test_save/test_newest_predict_sac_2.ckpt", mode='predict')


        # 开启网格搜索的随即投影矩阵,开启并行的情况不会消耗很多时间,同时可以提高双随即投影的效果
        if USE_GRID_SEARCH:
            # env_c = copy.copy(env)
            # env_c.action_dim = env.ld_adaptor.high_dim
            # self.grid_search_for_init_s(hes_low_seed, act_dim, obs_dim, exp_gen_rpm_train, exp_gen_rpm_test, env_c)
            print("[GRID SEARCH] START GRID SEARCH!")
            grid_dict = self.grid_search_parallel(hes_low_seed)
            # 找出值最大的键值对
            max_key, max_value = max(grid_dict.items(), key=lambda item: item[1])
            ori_key = hes_low_seed
            ori_value = grid_dict[ori_key]
            print("[GRID SEARCH] BEST SEED AMONG [{0}, {1}) is {2}, rate = {3}".format(hes_low_seed-globalValue.GRID_SERACH_AMPL, hes_low_seed+globalValue.GRID_SERACH_AMPL+1, max_key, max_value))
            self.hes_low_seed = hes_low_seed = max_key
            print("[GRID SEARCH] CHANGE DEFAULT SEED TO {0}".format(hes_low_seed))
            env.ld_adaptor = LowDimAdaptor(None, globalValue.HES_LOW_ACT_DIM, hes_low_seed, env.ld_adaptor.high_dim)
            print("[GRID SEARCH] REGENERATE THE LD ADAPTOR!")

            if USE_DOUBLE_NETWORK:
                print("[GRID SEARCH][DOUBLE] START GRID SEARCH DOUBLE!")
                grid_dict = self.grid_search_parallel_for_double(hes_low_seed)
                # 找出值最大的键值对
                max_key_double, max_value_double = max(grid_dict.items(), key=lambda item: item[1])
                print("[GRID SEARCH][DOUBLE] BEST SEED AMONG [{0}, {1}) is {2}, rate = {3}".format(hes_low_seed-globalValue.GRID_SERACH_AMPL, hes_low_seed+globalValue.GRID_SERACH_AMPL+1, max_key, max_value_double))
                self.hes_low_seed_double = hes_low_seed_double = max_key_double
                print("[GRID SEARCH][DOUBLE] CHANGE DEFAULT SEED TO {0}".format(hes_low_seed_double))
                env.ld_adaptor_double = LowDimAdaptor(None, globalValue.HES_LOW_ACT_DIM, hes_low_seed_double, env.ld_adaptor.high_dim)
                env.ld_adaptor_double.double_network_correct(env.ld_adaptor.A)
                env.ld_adaptor_double.double_network_correct(env.ld_adaptor.A)
                print("[GRID SEARCH][DOUBLE] REGENERATE THE LD ADAPTOR DOUBLE!")
                if hes_low_seed_double == 40:
                    print("[GRID SERACH DEBUG][AFTER CHANGE] env.ld_adaptor.A = ", env.ld_adaptor.A)
                    print("[GRID SERACH DEBUG][AFTER CHANGE] env.ld_adaptor_double.A = ", env.ld_adaptor_double.A)

            print("[GRID SEARCH][INFO][200 episode] origin key = {0}, rate = {1}".format(ori_key, ori_value))
            print("[GRID SEARCH][INFO][200 episode] single key = {0}, rate = {1}".format(max_key, max_value))
            print("[GRID SEARCH][INFO][200 episode] double key = {0}, rate = {1}".format(max_key_double, max_value_double))

            

        # 打开存储信息的文件，准备写入
        f_mode = 'w+'
        f_step_reward = open("./test_model/scores.txt", f_mode, encoding="utf-8")
        f_time_store = open("./test_model/timestore.txt", f_mode, encoding="utf-8")

        f_qps_store = open("./test_model/qps_store.txt", f_mode, encoding="utf-8")
        f_qps_store.close()

        f_critc_loss = open("./test_model/critic_loss.txt", f_mode, encoding="utf-8")
        f_critc_loss.close()
        f_pr = open("./test_model/PCA&RF.txt", f_mode, encoding="utf-8")
        f_pr.close()

        if env.info == 'CE':
            f_bp_size = open("./test_model/buffer_pool_size.txt", f_mode, encoding="utf-8")
            f_bp_size.close()
            f_hit_ratio = open("./test_model/hit_ratio.txt", f_mode, encoding="utf-8")
            f_hit_ratio.close()
        if env.info == 'NODES':
            f_bp_size_se = open("./test_model/buffer_pool_size_se.txt", f_mode, encoding="utf-8")
            f_bp_size_se.close()
            f_hit_ratio_se = open("./test_model/hit_ratio_se.txt", f_mode, encoding="utf-8")
            f_hit_ratio_se.close()
            f_bp_size_ce = open("./test_model/buffer_pool_size_ce.txt", f_mode, encoding="utf-8")
            f_bp_size_ce.close()
            f_hit_ratio_ce = open("./test_model/hit_ratio_ce.txt", f_mode, encoding="utf-8")
            f_hit_ratio_ce.close()
            f_best_action = open("./test_model/best_action.log", f_mode, encoding="utf-8")
            f_best_action.close()
            # f_fl_train_loss = open("./test_model/fl_train_loss.log", f_mode, encoding="utf-8")
            # f_fl_train_loss.close()
            # f_fl_time = open("./test_model/fl_time.log", f_mode, encoding="utf-8")
            # f_fl_time.close()

        if os.path.exists('./test_model/bestnow.log'):
            os.remove('./test_model/bestnow.log')
        if os.path.exists('./test_model/bestall.log'):
            os.remove('./test_model/bestall.log')

        f_human_exp_hit = open("./test_model/human_exp_hit.txt", f_mode, encoding="utf-8")

        # 经验池预热时，需要写入actions原始值及对应reward到文件中
        # 随机森林使用到的原始数据(actions,reward)
        # 先写表头
        header = list(env.all_actions.keys())
        header.append('reward')
        with open('./test_model/actions_reward.csv', 'w', encoding='utf-8', newline='') as file_obj:
            # 1:创建writer对象
            writer = csv.writer(file_obj)
            # 2:写表头
            writer.writerow(header)

        # 为状态空间降维，特征提取
        new_components = PCA_REMAIN
        self.pca = pca = PCA(n_components=new_components)

        expr_name = 'train_{}_{}_{}'.format(env.info, env.method, str(utils.time_to_str(utils.get_timestamp())))
        self.TD3_logger = TD3_logger = utils.Logger(
            name=env.method,
            log_file='./test_model/log/{}.log'.format(expr_name)
        )
        
        rpm = None
        rpm_double = None
        # 创建经验池

        ld = act_dim
        hd = act_dim
        if USE_LD_ADAPTOR:
            ld = env.ld_adaptor.target_dim
            hd = env.ld_adaptor.high_dim
        self.rpm = rpm = ReplayMemory(MEMORY_SIZE, act_dim, obs_dim, USE_LD_ADAPTOR, ld, hd)
        if USE_DOUBLE_NETWORK:
            self.rpm_double = rpm_double = ReplayMemory(MEMORY_SIZE, act_dim, obs_dim, USE_LD_ADAPTOR, ld, hd)
        
        self.rpm_full_real = ReplayMemory(MEMORY_SIZE, act_dim, obs_dim, USE_LD_ADAPTOR, ld, hd)


        
        fl_start_time = time.time()
        # 趋势快速学习的主要函数
        if USE_EXP_GEN_FAST_LERAN and (0 in EXP_GEN_FAST_LERAN_STAGE):
            
            # 测一下最开始的loss和一致性
            for i in range(1):
                (exp_batch_obs, exp_batch_action, exp_batch_reward, exp_batch_next_obs, exp_batch_done, exp_batch_last_action, exp_batch_trend) = \
                            exp_gen_rpm_test.exp_fast_learn_sample(EXP_GEN_LEARN_BATCH_SIZE)
                # n_exp_batch_obs = exp_batch_obs 
                n_exp_batch_obs = self.agent.normalizerBatch(exp_batch_obs, exp_gen_rpm_test)
                
                sum_hit_cnt = 0
                for j in range(EXP_GEN_LEARN_BATCH_SIZE):
                    
                    pred_action = self.agent.predict(n_exp_batch_obs[j], np.array([exp_batch_last_action[j]]).astype('float32'))
                    if USE_DOUBLE_NETWORK:
                        pred_action_double = self.agent_double.get_trend(n_exp_batch_obs[j], np.array([exp_batch_last_action[j]]).astype('float32'))

                    last_action = exp_batch_last_action[j]
                    exp_trend = exp_batch_trend[j]
                    pred_action = pred_action - 1.0
                    exp_trend = exp_trend - 1.0
                    
                    if USE_LD_ADAPTOR:
                        pred_action = env.ld_adaptor.transform(pred_action)
                        if USE_DOUBLE_NETWORK:
                            pred_action_double = env.ld_adaptor_double.transform(pred_action_double)

                        # last_action = env.ld_adaptor.transform(last_action)
                        # exp_trend = env.ld_adaptor.transform(exp_trend)
                        
                    # print("[MERGE DEBUG] pred_action shape = ", pred_action.shape)
                    # print("[MERGE DEBUG] pred_action = ", pred_action)
                    
                    # print("[MERGE DEBUG] last_action shape = ", last_action.shape)
                    # delta_action = pred_action - last_action
                    delta_action = pred_action
                    if USE_DOUBLE_NETWORK:
                        # delta_action = (pred_action + pred_action_double) * 0.5
                        delta_action = pred_action * globalValue.DOUBLE_RATIO + pred_action_double * (1-globalValue.DOUBLE_RATIO)

                    sign_delta_action = np.sign(delta_action)
                    product_sign = np.multiply(sign_delta_action, exp_trend)
                    # print("before sign: ", product_sign)
                    product_sign = np.sign(product_sign)
                    # print("after sign: ", product_sign)
                    
                    hit_cnt = np.sum(product_sign, axis=0)
                    abs_cnt = np.sum(np.abs(product_sign), axis=0)
                    hit_r_single = hit_cnt * 1.0 / abs_cnt
                    # print("[FL DUBEG] hit_r_single = ", hit_r_single)
                    if hit_r_single > EXP_GEN_HIT_THRES:
                        
                        sum_hit_cnt += 1
                            
                print("[FAST LEARN init] evaluate USE_EXP_GEN_FAST_LERAN use gen test, rate \t= \t{0}".format(sum_hit_cnt / EXP_GEN_LEARN_BATCH_SIZE))
            
            
            if EXP_GEN_USE_ACTUAL_ENV_TEST:
                for i in range(3):
                    # (exp_batch_obs, exp_batch_action, exp_batch_reward, exp_batch_next_obs, exp_batch_done, exp_batch_last_action) = \
                    #             exp_gen_rpm_actual_env.sample(EXP_GEN_ACTUAL_ENV_SAMPLE_NUM)
                    (exp_batch_obs, exp_batch_action, exp_batch_reward, exp_batch_next_obs, exp_batch_done, exp_batch_last_action, mini_batch_idx) = \
                                exp_gen_rpm_actual_env.sample_for_pca(EXP_GEN_ACTUAL_ENV_SAMPLE_NUM)
                    (exp_batch_obs_origin, exp_batch_action_origin, exp_batch_reward_origin, exp_batch_next_obs_origin, exp_batch_done_origin, exp_batch_last_action_origin) = \
                                exp_gen_rpm_actual_env.sample_for_fix(mini_batch_idx)
                    exp_batch_trend = env.lerc.get_trend(exp_batch_obs, env)
                    # print("exp_batch_trend = ", exp_batch_trend)
                    # n_exp_batch_obs = exp_batch_obs
                    n_exp_batch_obs = self.agent.normalizerBatch(exp_batch_obs, exp_gen_rpm_actual_env)
                    
                    sum_hit_cnt = 0
                    for j in range(EXP_GEN_ACTUAL_ENV_SAMPLE_NUM):
                                            
                        last_action = np.array([exp_batch_last_action[j]]).astype('float32')                    
                        
                        # if USE_LD_ADAPTOR:
                        #     last_action = env.ld_adaptor.reverse_transform(np.transpose(last_action)).astype(np.float32)
                        #     last_action = np.transpose(last_action).astype(np.float32)
                        
                        pred_action = self.agent.predict(n_exp_batch_obs[j], last_action).astype(np.float32)
                        if USE_DOUBLE_NETWORK:
                            pred_action_double = self.agent_double.get_trend(n_exp_batch_obs[j], last_action).astype(np.float32)
                        # last_action = exp_batch_last_action[j]
                        exp_trend = exp_batch_trend[j]
                        pred_action = pred_action - 1.0
                        exp_trend = exp_trend - 1.0
                        
                        if USE_LD_ADAPTOR:
                            pred_action = env.ld_adaptor.transform(pred_action)
                            if USE_DOUBLE_NETWORK:
                                pred_action_double = env.ld_adaptor_double.transform(pred_action_double)
                            # last_action = env.ld_adaptor.transform(np.transpose(last_action).astype(np.float32))
                            # last_action = last_action.reshape(last_action.shape[0])
                            # exp_trend = exp_trend
                        else:
                            last_action = last_action.reshape(last_action.shape[1])
                        
                        
                        # print("[MERGE DEBUG] pred_action shape = ", pred_action.shape)
                        # print("[MERGE DEBUG] last_action shape = ", last_action.shape)
                        # delta_action = pred_action - last_action
                        delta_action = pred_action
                        if USE_DOUBLE_NETWORK:
                            # delta_action = (pred_action + pred_action_double) * 0.5
                            delta_action = pred_action * globalValue.DOUBLE_RATIO + pred_action_double * (1-globalValue.DOUBLE_RATIO)
                        
                        # print("[MERGE DEBUG] delta_action shape = ", delta_action.shape)

                        # pre_trend = self.agent.alg.get_pred_trend(n_exp_batch_obs[j], np.array([exp_batch_last_action[j]]).astype('float32'))
                        
                        
                        sign_delta_action = np.sign(delta_action)
                        
                        product_sign = np.multiply(sign_delta_action, exp_trend)
                        product_sign = np.sign(product_sign)
                        # hit_cnt = np.sum(product_sign, axis=0)
                        # # print("[MERGE DEBUG] product_sign shape = ", product_sign.shape)
                        # if hit_cnt > 0:
                        #     sum_hit_cnt += 1
                        hit_cnt = np.sum(product_sign, axis=0)
                        abs_cnt = np.sum(np.abs(product_sign), axis=0)
                        hit_r_single = hit_cnt * 1.0 / abs_cnt
                        # print("[FL DUBEG] hit_r_single = ", hit_r_single)
                        if hit_r_single > EXP_GEN_HIT_THRES:
                            sum_hit_cnt += 1
                    # print("sum_hit_cnt = ", sum_hit_cnt) 
                    print("[FAST LEARN init] evaluate USE_EXP_GEN_FAST_LERAN use actual env test, rate \t= \t{0}".format(sum_hit_cnt / EXP_GEN_ACTUAL_ENV_SAMPLE_NUM))
                
            
            
            f_fl_train_loss = open("./test_model/fl_train_loss.log", 'w+', encoding="utf-8")
            
            
            
            print("\n=========================================")
            total_loss = 0
            loss_1 = 0
            loss_2 = 0
            test_continue_num = 3
            delta_step = 1000
            # 学习EXP_GEN_LEARN_EPISODE轮,
            for i in range(EXP_GEN_LEARN_EPISODE):
                # for j in range(EXP_GEN_LEARN_BATCH_SIZE):
                (exp_batch_obs, exp_batch_action, exp_batch_reward, exp_batch_next_obs, exp_batch_done, exp_batch_last_action, exp_batch_trend) = \
                    exp_gen_rpm_train.exp_fast_learn_sample(EXP_GEN_LEARN_BATCH_SIZE)
            
                # n_exp_batch_obs = exp_batch_obs 
                # n_exp_batch_next_obs = exp_batch_next_obs 
                n_exp_batch_obs = self.agent.normalizerBatch(exp_batch_obs, exp_gen_rpm_train)
                n_exp_batch_next_obs = self.agent.normalizerBatch(exp_batch_next_obs, exp_gen_rpm_train)
                # print("[MERGE DEBUG] exp_batch_trend.shape = ", exp_batch_trend.shape)
                # print("[MERGE DEBUG] exp_batch_last_action.shape = ", exp_batch_last_action.shape)
                # print("[MERGE DEBUG] exp_batch_trend.shape = ", exp_batch_trend.shape)
                A = []
                if USE_LD_ADAPTOR:
                    # A = env.ld_adaptor.A
                    A = np.stack([env.ld_adaptor.A] * EXP_GEN_LEARN_BATCH_SIZE, axis=0).astype(np.float32)
                else:
                    # A = np.eye(env.action_dim)
                    A = np.stack([np.eye(env.action_dim)] * EXP_GEN_LEARN_BATCH_SIZE, axis=0).astype(np.float32)
                # print("[MERGE DEBUG] A shape = ", A.shape)
                # loss = self.agent.exp_fast_learn(n_exp_batch_obs, exp_batch_last_action, exp_batch_trend, A)
                # total_loss += loss
                
                if USE_DOUBLE_NETWORK:
                    A_1 = A
                    A_2 = []
                    A_2 = np.stack([env.ld_adaptor_double.A] * EXP_GEN_LEARN_BATCH_SIZE, axis=0).astype(np.float32)
                    exp_batch_trend_normal = exp_batch_trend - 1
                    # print("[DOUBLE NETWORK DEBUG] exp_batch_trend_normal shape = ", exp_batch_trend_normal.shape)
                    # print("[DOUBLE NETWORK DEBUG] exp_batch_trend_normal = ", exp_batch_trend_normal)
                    batch_net_1_trend = self.agent.get_trend(n_exp_batch_obs, np.array([exp_batch_last_action]).astype('float32')) - 1
                    batch_net_2_trend = self.agent_double.get_trend(n_exp_batch_obs, np.array([exp_batch_last_action]).astype('float32'))
                    # print("[DOUBLE NETWORK DEBUG] batch_net_1_trend shape = ", batch_net_1_trend.shape)
                    # print("[DOUBLE NETWORK DEBUG] batch_net_1_trend = ", batch_net_1_trend)
                    # print("[DOUBLE NETWORK DEBUG] batch_net_2_trend shape = ", batch_net_2_trend.shape)
                    # print("[DOUBLE NETWORK DEBUG] batch_net_2_trend = ", batch_net_2_trend)
                    batch_net_1_trend_hd = env.ld_adaptor.batch_transform(batch_net_1_trend)
                    batch_net_2_trend_hd = env.ld_adaptor_double.batch_transform(batch_net_2_trend)
                    # print("[DOUBLE NETWORK DEBUG] batch_net_1_trend_hd shape = ", batch_net_1_trend_hd.shape)
                    # print("[DOUBLE NETWORK DEBUG] batch_net_2_trend_hd shape = ", batch_net_2_trend_hd.shape)
                    # print("[DOUBLE NETWORK DEBUG] batch_net_1_trend_hd = ", batch_net_1_trend_hd)
                    # print("[DOUBLE NETWORK DEBUG] batch_net_2_trend_hd = ", batch_net_2_trend_hd)
                    batch_target_trend_1 = 2 * exp_batch_trend_normal - batch_net_2_trend_hd
                    batch_target_trend_2_minus = (exp_batch_trend_normal - globalValue.DOUBLE_RATIO * batch_net_1_trend_hd) / (1-globalValue.DOUBLE_RATIO)
                    batch_target_trend_2_sign = np.sign(batch_target_trend_2_minus)
                    batch_target_trend_2 = (np.abs(batch_target_trend_2_minus) + 0.2) * np.abs(np.sign(exp_batch_trend_normal)) * batch_target_trend_2_sign
                    # batch_target_trend_2 = batch_target_trend_2_minus
                    # print("[DOUBLE NETWORK DEBUG] batch_target_trend_1 shape = ", batch_target_trend_1.shape)
                    # print("[DOUBLE NETWORK DEBUG] batch_target_trend_2 shape = ", batch_target_trend_2.shape)
                    # print("[DOUBLE NETWORK DEBUG] batch_target_trend_1 = ", batch_target_trend_1)
                    # print("[DOUBLE NETWORK DEBUG] batch_target_trend_2 = ", batch_target_trend_2)
                    batch_target_label_1 = np.sign(batch_target_trend_1) + 1
                    # batch_target_label_2 = np.sign(batch_target_trend_2) + 1
                    batch_target_label_2 = batch_target_trend_2
                                        
                    loss = self.agent.exp_fast_learn(n_exp_batch_obs, exp_batch_last_action, exp_batch_trend, A_1)
                    loss_double = self.agent_double.exp_fast_learn_double(n_exp_batch_obs, exp_batch_last_action, batch_target_label_2, A_2)
                else:
                    loss = self.agent.exp_fast_learn(n_exp_batch_obs, exp_batch_last_action, exp_batch_trend, A)
                
                
                if USE_DOUBLE_NETWORK:
                    # total_loss += (loss+loss_double)*0.5
                    loss_1 += loss
                    loss_2 += loss_double
                    total_loss += loss * globalValue.DOUBLE_RATIO + loss_double * (1-globalValue.DOUBLE_RATIO)
                    f_fl_train_loss.write("{0}, {1}, {2}\n".format(loss_1 * 1.0 / (i+1), loss_2 * 1.0 / (i+1),  total_loss * 1.0 / (i+1)))
                else:
                    total_loss += loss
                    f_fl_train_loss.write("{0}\n"(total_loss * 1.0 / (i+1)))
                
                
                if i % (delta_step/10) == 0:
                    if USE_DOUBLE_NETWORK:
                        print("[FAST LEARN {0}] loss_1 = {1}, loss_2 = {2}, total_loss = {3}".format(i, loss_1 * 1.0 / (i+1), loss_2 * 1.0 / (i+1),  total_loss * 1.0 / (i+1)))
                    else:
                        print("[FAST LEARN {0}] loss = {1}".format(i, total_loss * 1.0 / (i+1)))

                # 测试
                
                if i % delta_step == 0:
                    (exp_batch_obs, exp_batch_action, exp_batch_reward, exp_batch_next_obs, exp_batch_done, exp_batch_last_action, exp_batch_trend) = \
                        exp_gen_rpm_test.exp_fast_learn_sample(EXP_GEN_LEARN_BATCH_SIZE)
                    # n_exp_batch_obs = exp_batch_obs
                    n_exp_batch_obs = self.agent.normalizerBatch(exp_batch_obs, exp_gen_rpm_test)
                    
                    sum_hit_cnt = 0
                    total_hit_ratio_single = 0.0
                    for j in range(EXP_GEN_LEARN_BATCH_SIZE):
                        
                        pred_action = self.agent.get_trend(n_exp_batch_obs[j], np.array([exp_batch_last_action[j]]).astype('float32'))
                        if USE_DOUBLE_NETWORK:
                            pred_action_double = self.agent_double.get_trend(n_exp_batch_obs[j], np.array([exp_batch_last_action[j]]).astype('float32'))

                        last_action = exp_batch_last_action[j]
                        exp_trend = exp_batch_trend[j]
                        exp_trend = exp_trend - 1.0
                        pred_action = pred_action - 1.0
                        
                        
                        
                        
                        if USE_LD_ADAPTOR:
                            pred_action = env.ld_adaptor.transform(pred_action)
                            if USE_DOUBLE_NETWORK:
                                pred_action_double = env.ld_adaptor_double.transform(pred_action_double)
                        
                        # print("[FL DUBEG] pred_action = ", (list)(pred_action))
                        # print("[FL DUBEG] exp_trend = ", (list)(exp_trend))
                        
                        delta_action = pred_action
                        if USE_DOUBLE_NETWORK:
                            delta_action = pred_action * globalValue.DOUBLE_RATIO + pred_action_double * (1-globalValue.DOUBLE_RATIO)
                        sign_delta_action = np.sign(delta_action)
                        product_sign = np.multiply(sign_delta_action, exp_trend)
                        product_sign = np.sign(product_sign)                        

                        hit_cnt = np.sum(product_sign, axis=0)
                        abs_cnt = np.sum(np.abs(product_sign), axis=0)
                        hit_r_single = hit_cnt * 1.0 / abs_cnt
                        # print("[FL DUBEG] hit_r_single = ", hit_r_single)
                        total_hit_ratio_single += hit_r_single
                        
                        if hit_r_single > EXP_GEN_HIT_THRES:
                            sum_hit_cnt += 1
                        
                    test_rate = sum_hit_cnt / EXP_GEN_LEARN_BATCH_SIZE
                    test_single_rate = total_hit_ratio_single / EXP_GEN_LEARN_BATCH_SIZE
                    print("[FL DUBEG] total_hit_ratio_single = ", total_hit_ratio_single)
                    print("[FAST LEARN {0}] evaluate USE_EXP_GEN_FAST_LERAN use gen test, rate = {1}, rate_single = {2}".format(i, sum_hit_cnt / EXP_GEN_LEARN_BATCH_SIZE, total_hit_ratio_single / EXP_GEN_LEARN_BATCH_SIZE))
                    
                   
                    actual_rate = 1.0
                    
                    
                    # 早停判断  
                    if test_rate >= EXP_GEN_GEN_TEST_RATE_THRESHOLD and actual_rate >= EXP_GEN_ACTUAL_ENV_RATE_THRESHOLD:
                        if test_rate > 0.99 and test_single_rate > 0.99:
                            break
                        if test_continue_num > 0:
                            test_continue_num -= 1
                            if test_continue_num == 0:
                                break
                        else:
                            break
                    else:
                        test_continue_num = 3
                    
            f_fl_train_loss.close()    
                    
            print("=========================================\n")
            print("finish USE_EXP_GEN_FAST_LERAN!")            

        fl_train_end_time = time.time()
        fl_train_time = fl_train_end_time - fl_start_time
        print("[FAST LEARN TIME] train_time = {:.4f} s".format(fl_train_time))
        with open("./test_model/fl_time.log", "a+", encoding="utf-8") as f_fl_time:
            f_fl_time.write("[FAST LEARN TIME] train_time = {:.4f} s\n".format(fl_train_time))
        
        

        src = globalValue.RPM_SRC
        dest = globalValue.RPM_DEST
        # 往经验池中预存数据
        while len(rpm) < MEMORY_WARMUP_SIZE:
            print("经验池中数据数量: ", len(rpm))
            
            total_reward, steps = self.run_episode(self.model, self.agent, env, rpm, pca, f_step_reward, TD3_logger, mode, self.rpm_full_real)
            f_time_store.write('warmup_time=' + str(env.end_time - env.start_time) + '\n')
            f_time_store.flush()
            f_human_exp_hit.write('human_exp_hit=' + str(env.human_exp_hitcnt * 1.0 / (steps+0.0001)) + '\n')
            f_human_exp_hit.flush()

            
                                                                                        

        # 迁移的主要代码,使用历史经验供模型学习
        if USE_HIS_RPM:
            his_file_name = HIS_TRANSFER_RPM_PATH
            his_rpm_file = open(his_file_name, "rb")
            his_rpm = pickle.load(his_rpm_file)
            
            his_double_file_name = HIS_TRANSFER_RPM_DOUBLE_PATH
            his_rpm_double_file = open(his_double_file_name, "rb")
            his_rpm_double = pickle.load(his_rpm_double_file)
            
            for (batch_obs, batch_action, batch_reward, batch_next_obs, batch_done, batch_last_action) in his_rpm.buffer:
                batch_action = [np.array(batch_action).astype(np.float32)]
                batch_last_action = [batch_last_action]
                if batch_done == 0.0:
                    batch_done = False
                else:
                    batch_done = True
                rpm.append((batch_obs, batch_action, batch_reward, batch_next_obs, batch_done, batch_last_action))
                
            for (batch_obs, batch_action, batch_reward, batch_next_obs, batch_done, batch_last_action) in his_rpm_double.buffer:
                batch_action = [np.array(batch_action).astype(np.float32)]
                batch_last_action = [batch_last_action]
                if batch_done == 0.0:
                    batch_done = False
                else:
                    batch_done = True
                rpm_double.append((batch_obs, batch_action, batch_reward, batch_next_obs, batch_done, batch_last_action))
            
            
            if  env.method == 'SAC' or env.method == 'SAC_2':
                #if USE_PRIORITY_RPM:
                    #new_td_error = self.agent.cal_td_error(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)
                    #print("*************************************")
                    #print("*new_td_error_before_learn = {0}".format(new_td_error))
                    #print("*************************************")
                
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("[TRANSFER LEARN]agent learn, alg = {0}, time = pretrain_epsisode".format(env.method))
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

                # print(batch_action)
                all_his_len = len(rpm.buffer)
                
                for transfer_i in range(HIS_TRANSFER_ITER):
                    
                    his_learn_batch_idxs = random.sample(range(0, all_his_len), HIS_TRANSFER_BATCH)
                    (batch_obs, batch_action, batch_reward, batch_next_obs, batch_done, batch_last_action) = rpm.sample_for_transfer(his_learn_batch_idxs)
                    batch_obs = self.agent.normalizerBatch(batch_obs, rpm)
                    batch_next_obs = self.agent.normalizerBatch(batch_next_obs, rpm)
                    
                    (batch_obs_d, batch_action_d, batch_reward_d, batch_next_obs_d, batch_done_d, batch_last_action_d) = rpm_double.sample_for_transfer(his_learn_batch_idxs)
                    batch_obs_d = self.agent_double.normalizerBatch(batch_obs_d, rpm_double)
                    batch_next_obs_d = self.agent_double.normalizerBatch(batch_next_obs_d, rpm_double)
                    
                    resize_batch_reward = batch_reward.copy()
                    critic_cost = 0
                    actor_cost, critic_cost = self.agent.learn(batch_obs, batch_action, resize_batch_reward, batch_next_obs,
                                                        batch_done, batch_last_action)
                    
                        
                        
                    if USE_DOUBLE_NETWORK:                        
                        batch_action_double = np.array(batch_action_d)
                        actor_cost_double, critic_cost_double = self.agent_double.learn(batch_obs, batch_action_double, batch_reward, batch_next_obs,
                                                        batch_done, batch_last_action)
                        
                    if USE_DOUBLE_NETWORK:
                        if transfer_i % HIS_TRANSFER_PRINT_ITER == 0:
                            print("[TRANSFER LEARN][{0}]\tac = {1:.4f},\tcc = {2:.4f},\tac_d = {3:.4f},\tcc_d = {4:.4f}"
                                .format(transfer_i, actor_cost, critic_cost, actor_cost_double, critic_cost_double))
                    else:
                        if transfer_i % HIS_TRANSFER_PRINT_ITER == 0:
                            print("[TRANSFER LEARN] ac = {0},\tcc = {1}".format(actor_cost, critic_cost))

                if not USE_PRIORITY_RPM:
                    ld = act_dim
                    hd = act_dim
                    if USE_LD_ADAPTOR:
                        ld = env.ld_adaptor.target_dim
                        hd = env.ld_adaptor.high_dim
                    self.rpm = rpm = ReplayMemory(MEMORY_SIZE, act_dim, obs_dim, USE_LD_ADAPTOR, ld, hd)
                    if USE_DOUBLE_NETWORK:
                        self.rpm_double = rpm_double = ReplayMemory(MEMORY_SIZE, act_dim, obs_dim, USE_LD_ADAPTOR, ld, hd)
                    
                    self.rpm_full_real = ReplayMemory(MEMORY_SIZE, act_dim, obs_dim, USE_LD_ADAPTOR, ld, hd)
                
                else:
                    self.rpm = rpm = PrioritizedReplayMemory(MEMORY_SIZE, act_dim, obs_dim)
            
            
            # self.rpm = rpm =  his_rpm


        print('===>经验池预热完成!')
        print("===>经验池中数据数量: ", len(rpm))
        if USE_DOUBLE_NETWORK:
            print("===>经验池中数据数量: ", len(rpm_double))

        ckpt = './test_model/test_save/first_phase_{}'.format(str(utils.time_to_str(utils.get_timestamp())))
        f = open(ckpt + "_nodes_rpm.txt", "wb")
        # 保存回放内存,写盘很费时间，注意控制写盘频率
        pickle.dump(rpm, f)
        f.close()
        print('save rpm ok')
        # 预热完成修改env状态
        env.state = 1
        # reset the variables in WARMUP
        flag = True
        while flag:
            obs, flag = env.reset()


        # 训练模型

        f_train_reward = open("./test_model/train_reward_cal.txt", f_mode, encoding="utf-8")
        f_eval_reward = open("./test_model/eval_reward_cal.txt", f_mode, encoding="utf-8")

        episode = 0
        if env.expert_exp == True:
            env.action_trend_choice = ACTION_TREND_P
        env.best_action_choice = BEST_NOW_P

        # 强化学习与环境交互的过程
        while episode < TRAIN_EPISODE:
            # 每训练5个episode，做一次评估
            print('Start a new round,this round include %d episode and 1 evaluate process!' % EVAL_INTERVAL)
            for i in range(EVAL_INTERVAL):
                print('=========>>> episode = ', episode)
                total_reward, steps = self.run_episode(self.model, self.agent, env, rpm, pca, f_step_reward, TD3_logger, self.rpm_full_real, rpm_double)

                if env.action_trend_choice < 0.1:
                    env.action_trend_choice = 0.1
                f_train_reward.write("train_reward="+str(total_reward)+"\n")
                f_train_reward.flush()
                f_time_store.write('episode_time='+str(env.end_time - env.start_time)+"\n")
                f_time_store.flush()
                f_human_exp_hit.write('human_exp_hit='+str(env.human_exp_hitcnt * 1.0 / steps)+'\n')
                f_human_exp_hit.flush()
                print('Episode:{}    Test reward:{}'.format(episode, total_reward))
                episode += 1
                # if flag == 0:
                #     f = open("./1se/rpm_dir/se_rpm_new.txt", "wb")
                # elif flag == 1:
                #     f = open("./test_model/test_save/ce_rpm_new.txt", "wb")
                # else:
                #     f = open("./test_model/test_save/nodes_rpm_new.txt", "wb")
                #     # 保存回放内存,写盘很费时间，注意控制写盘频率
                # print('-----------save_rpm-----------')
                # pickle.dump(rpm, f)
                # f.close()
                
                # ckpt = './test_model/test_save/full_real_{}'.format(str(utils.time_to_str(utils.get_timestamp())))
                # f = open(ckpt + "_rpm.txt", "wb")
                # # 保存回放内存,写盘很费时间，注意控制写盘频率
                # pickle.dump(self.rpm_full_real, f)
                # f.close()
                # print('save rpm ok')
                
                # SAC
                # 保存模型        
                ckpt = './test_model/test_save/test_newest'
                print('-----------save_model-----------')
                print("[TRANSFER] self.hes_low_seed = {0}, self.hes_low_seed_double = {1}".format(self.hes_low_seed, self.hes_low_seed_double))
                print("[TRANSFER] globalValue.HES_LOW_ACT_DIM = {0}, globalValue.HES_LOW_ACT_DIM_DOUBLE = {1}".format(globalValue.HES_LOW_ACT_DIM, globalValue.HES_LOW_ACT_DIM_DOUBLE))
                print('ckpt = ', ckpt)
                if env.method == 'SAC_2':
                    
                    print("[TRANSFER][DENUG][MODEL] ", self.agent.alg.actor.act_dim)
                    print("[TRANSFER][DENUG][AGENT] ", self.agent.obs_dim)
                    print("[TRANSFER][DENUG][AGENT] ", self.agent.low_act_dim)
                    print("[TRANSFER][DENUG][AGENT] ", self.agent.high_act_dim)
                    
                    
                    self.agent.save(save_path=ckpt + '_predict_sac_2.ckpt', mode='predict')
                    self.agent.save(save_path=ckpt + '_train_learn_sac_2.ckpt', mode='learn')
                    if USE_DOUBLE_NETWORK:
                        self.agent_double.save(save_path=ckpt + '_predict_sac_2.ckpt', mode='predict')
                        self.agent_double.save(save_path=ckpt + '_train_learn_sac_2.ckpt', mode='learn')
                
                
                if globalValue.USE_MAKE_TRANSFER_DATA:
                    folder_path = os.path.join("./test_model/transfer_data", globalValue.TRANSFER_DATA_PATH)
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                        print(f"[TRANSFER_DATA NOTE] Folder '{folder_path}' created successfully.")
                    else:
                        print(f"[TRANSFER_DATA NOTE] Folder '{folder_path}' already exists.")
                    
                    rpm_transfer = ReplayMemory(100000000, env.ld_adaptor.target_dim, env.state_dim, True, env.ld_adaptor.target_dim, env.ld_adaptor.high_dim)
                    rpm_transfer_double = ReplayMemory(100000000, env.ld_adaptor.target_dim, env.state_dim, True, env.ld_adaptor.target_dim, env.ld_adaptor.high_dim)
    
                    transfer_info = {
                        "meta": {
                            "default":{
                                "sim": 1
                            }
                        },
                        "his_agent_dict": {"default": self.agent},
                        "his_agent_double_dict": {"default": self.agent_double},
                        "his_ld_dict": {"default": env.ld_adaptor},
                        "his_ld_double_dict": {"default": env.ld_adaptor_double},
                        "ld": env.ld_adaptor.target_dim,
                        "hd": env.ld_adaptor.high_dim,
                        "obs_dim": env.state_dim,
                        "node_num": len(globalValue.CONNECT_SE_IP) + len(globalValue.CONNECT_CE_IP),
                        "init_state": env.init_state_for_transfer,
                        "init_ratio": 0.2
                    }
                    print("[TRANSFER_DATA NOTE] env.state_dim = ", env.state_dim)
                    make_model.generate_transfer_exp_use_model(globalValue.TRANSFER_DATA_NUM, rpm_transfer, rpm_transfer_double, transfer_info)
                    print("[TRANSFER_DATA NOTE] rpm len = ", len(rpm.buffer))
                    print("[TRANSFER_DATA NOTE] rpm_double len = ", len(rpm_double.buffer))
                    print("[TRANSFER_DATA NOTE] rpm[0] = ", rpm.buffer[0])
                    print("[TRANSFER_DATA NOTE] rpm_double[0] = ", rpm_double.buffer[0])
                    
                    rpm_file_name = os.path.join(folder_path, "rpm_episode_{0}.txt".format(episode))
                    rpm_double_file_name = os.path.join(folder_path, "rpm_double_episode_{0}.txt".format(episode))
                    f_rpm = open(rpm_file_name, "wb")    
                    pickle.dump(rpm_transfer, f_rpm)
                    f_rpm.close()
                    print("[TRANSFER_DATA NOTE] rpm saved at ", rpm_file_name)
                    f_rpm_double = open(rpm_double_file_name, "wb")    
                    pickle.dump(rpm_transfer_double, f_rpm_double)
                    f_rpm_double.close()
                    print("[TRANSFER_DATA NOTE] rpm_double saved at ", rpm_double_file_name)
                    
                    
            print('-------------start_eval_test-------------')

            eval_reward = self.evaluate_ce(env, self.agent, rpm, pca, TD3_logger)
            f_eval_reward.write("eval_reward="+str(eval_reward)+"\n")
            f_eval_reward.flush()
            f_time_store.write('eval_time=' + str(env.end_time - env.start_time) + "\n")
            f_time_store.flush()
            print('Eval:{}    Test reward:{}'.format(env.eval, eval_reward))

        TD3_logger.end_time = TD3_logger.get_timestr()
        TD3_logger.info("ALL the episode done, start time = %s, end time = %s" % (TD3_logger.start_time, TD3_logger.end_time))

        # 关闭文件
        f_step_reward.close()
        f_train_reward.close()
        f_eval_reward.close()
        f_time_store.close()
        f_human_exp_hit.close()
        
        # 这段是保存经验池，暂时没啥用，代码可保留，之后参考写法
        # ckpt = './test_model/test_save/test_save_final_nodes_{}'.format(str(utils.time_to_str(utils.get_timestamp())))
        # f = open(ckpt + "_nodes_rpm_new.txt", "wb")
        # # 保存回放内存,写盘很费时间，注意控制写盘频率
        # print('-----------save_rpm_new-----------')
        # if TWO_PHASE == True:
        #     pickle.dump(rpm_new, f)
        # else:
        #     pickle.dump(rpm, f)
        # f.close()

        # # 保存模型
        # print('-----------save_model-----------')
        # print('ckpt = ', ckpt)

        # # SAC
        # if env.method == 'SAC':
        #     self.agent.save(save_path=ckpt + '_predict_sac.ckpt', mode='predict')
        #     self.agent.save(save_path=ckpt + '_train_learn_sac.ckpt', mode='learn')
            
        # # SAC
        # if env.method == 'SAC_2':
        #     self.agent.save(save_path=ckpt + '_predict_sac_2.ckpt', mode='predict')
        #     self.agent.save(save_path=ckpt + '_train_learn_sac_2.ckpt', mode='learn')



    



    def run_episode(self, model = None, agent = None, env = None, rpm = None, pca = None, f_step_reward = None, TD3_logger = None, mode = 1, rpm_full_real = None, rpm_double = None):
        
        if model == None:   model = self.model
        if agent == None:   agent = self.agent
        if env == None:     env = self.env
        if rpm == None:     rpm = self.rpm
        if pca == None:     pca = self.pca
        if f_step_reward == None:   f_step_reward = open("./test_model/scores.txt", "w+", encoding="utf-8")
        if TD3_logger == None:      TD3_logger = self.TD3_logger
        
        if rpm_full_real == None:   rpm_full_real = self.rpm_full_real
        if rpm_double == None:      rpm_double = self.rpm_double
            
            
        globalValue.EVAL_TEST = False
        obs = np.array([0])
        reset_val = False
        # if not USE_FIXED_DR_KNOBS or env.state == 1:
        reset_val = True
        env.start_time = time.time()
        while reset_val:
            obs, reset_val = env.reset()
        rear_obs = obs
        p_exp = env.action_trend_choice
        # 某轮总得分
        total_reward = 0
        reward_a = -1
        done = False
        steps = 0
        max_action = 1
        raw_action = 0
        action = []
        if env.state == 1:
            env.episode += 1
        accumulate_loss = 0


        s = '[' + env.method + 'Env initialized]'
        s += '[ ' + str(env.se_num) + ' ses:'
        for se in env.se_info:
            s += '{se' + str(se.uuid) + ', hit_ratio: ' + str(se.hit_t0) + ', buffer_pool_size: ' + str(se.bp_size_t0) + '}'
        s += '][ ' + str(env.ce_num) + ' ces:'
        for ce in env.ce_info:
            if ce.is_primary == True:
                s += '{ce' + str(ce.uuid) + ', qps: ' + str(ce.qps_t0) + ', hit_ratio: ' + str(ce.hit_t0 )+ ', buffer_pool_size: ' + str(ce.bp_size_t0) + '}'
            else:
                s += '{ce' + str(ce.uuid) + ', hit_ratio: ' + str(
                    ce.hit_t0) + ', buffer_pool_size: ' + str(ce.bp_size_t0) + '}'
        s += ']'
        TD3_logger.info("\n{}".format(s))


        # 每个episode里面有多个step
        # step和episode的区别是: step的修改是基于上一个step的配置和状态, episode是基于初始默认配置和状态
        # 即每个epsidode会重置一遍调优的起点,然后重新统计一遍默认的qps,这样也有助于减小性能波动带来的影响
        while steps < MOVE_STEPS:

            steps += 1
            if env.state == 0:
                print('==========>WARMUP-step', steps)
            elif env.state == 1:
                print('==========>Episode{}-step{} '.format(env.episode, steps))
            else:
                print('==========>Eval-step', steps)

            batch_obs = obs
            if done:
                print('+++ reset step +++')
                batch_obs = rear_obs
                obs = rear_obs
            done = False
            rear_obs = obs

            # 输入agent时对state归一化
            # print('batch_obs = ', batch_obs)
            input_obs = agent.normalizer(batch_obs, rpm)
            print("Normalize obs:", input_obs)
           
            if env.method == 'SAC' or env.method == 'SAC_2':
                print("[LD ADAPTOR] env.all_last_action = ", env.all_last_action)
                # action_pred = agent.sample(input_obs.astype('float32'), np.array([env.all_last_action]).astype('float32'))
                action_pred = agent.predict(input_obs.astype('float32'), np.array([env.all_last_action]).astype('float32'))
                if USE_DOUBLE_NETWORK:
                    action_trend_1 = agent.get_trend(input_obs.astype('float32'), np.array([env.all_last_action]).astype('float32'))-1.0
                    action_trend_2 = self.agent_double.get_trend(input_obs.astype('float32'), np.array([env.all_last_action]).astype('float32'))
                    action_ampl_t = agent.get_ampl(input_obs.astype('float32'), np.array([env.all_last_action]).astype('float32'))
                    action_1 = np.multiply(action_trend_1, action_ampl_t)
                    action_2 = np.multiply(action_trend_2, action_ampl_t)
                    # print("[DOUBLE NETWORK MERGE] action_trend_1 = ", action_trend_1)
                    # print("[DOUBLE NETWORK MERGE] action_trend_2 = ", action_trend_2)
                    # print("[DOUBLE NETWORK MERGE] action_ampl_t = ", action_ampl_t)
                    # print("[DOUBLE NETWORK MERGE] action_1 = ", action_1)
                    # print("[DOUBLE NETWORK MERGE] action_2 = ", action_2)

                    
                    
                print("[MERGE DEBUG] SAC pridect delta action = ", action_pred)
                

                                
                action = action_pred
                action_add = utils.add_delta_ation_to_last_action(np.array([env.all_last_action]).astype('float32'), action_pred, env.ld_adaptor)
                if USE_DOUBLE_NETWORK:
                    action_add = utils.add_delta_ation_to_last_action_double(np.array([env.all_last_action]).astype('float32'), action_1, env.ld_adaptor, action_2, env.ld_adaptor_double)
                action = action_add[0]
                print("[LD ADAPTOR] action = ", action)

                                       
                    
            if len(env.best_action_record) <=1: 
                env.best_action_record = action
            if len(env.all_last_action) <=1:
                env.all_last_action = action
                
            if USE_DOUBLE_NETWORK and env.last_action_double is None:
                env.last_action_double = np.zeros(env.ld_adaptor_double.target_dim)
                

            print('action by clip: ', action)
            action_trend = []
            # 在策略网络的输出、best_action_now、expert_exp中概率选择，这时需要一个随机数，看落在哪个区间内，
            # print("mode = ", mode)
            
            if USE_EXP_GEN_FAST_LERAN or USE_HIS_RPM:
                
                action_trend0 = datautils.all_nodes_labels_to_action_trend(env)
                print("input_obs = {0}\nshape = {1}".format(input_obs, input_obs.shape))

                action_trend = env.lerc.get_trend(np.array([batch_obs]).astype('float32'), env)[0]
                
                print("action_trend0 = ", action_trend0)
                print("action_trend = ", action_trend)
                print("label_0 = ", env.se_info[0].labelA, env.se_info[0].labelB, env.se_info[0].labelC, env.ce_info[0].labelA, env.ce_info[0].labelB, env.ce_info[0].labelC)
                print("label = ", env.lerc.batch_labels)
                
                print("[MERGE DECUG] action shape = ", action.shape)
                # print("[MERGE DECUG] env.all_last_action shape = ", env.all_last_action.shape)
                
                print("[MERGE DECUG][cal_human_hit] action = ", action)
                print("[MERGE DECUG][cal_human_hit] env.all_last_action = ", env.all_last_action)
                print("[MERGE DECUG][cal_human_hit] action_trend = ", action_trend)
                if USE_LD_ADAPTOR and USE_DOUBLE_NETWORK:
                    action_trend_c = datautils.all_nodes_labels_to_action_trend(env)
                    action = utils.action_corrext_for_bps(env, action, 0, 1, action_trend_c)

                
                exp_hit_cnt = utils.cal_human_hit(action, action_trend, env.all_last_action)
                if exp_hit_cnt > 0:
                    env.exp_fast_learn_hit_cnt += 1
                exp_hit_rate = env.exp_fast_learn_hit_cnt / (0.001 + steps)
            
            if (USE_EXPERT_EXP and (env.state == 1 or mode == 2)):
                # 正式训练时考虑best_action_now
                if env.expert_exp == True:
                    print("[MERGE DECUG] lerc trend = ", action_trend)
                    action_trend = datautils.all_nodes_labels_to_action_trend(env)
                    print("[MERGE DECUG] rule trend = ", action_trend)

                    # 实现概率衰减
                    p_exp *= 0.995
                    print('human action pick_p: ', p_exp)
                    print('human action_trend: ', action_trend)
                    env.action_trend_choice *= 0.9995
                if USE_LD_ADAPTOR:
                    action, hit_cnt, p_rand = utils.action_with_knowledge_and_best_now(action, env.best_action_record, action_trend,
                                                                env.best_action_choice, p_exp, env.ld_adaptor.transform(env.all_last_action))
                else:
                    action, hit_cnt, p_rand = utils.action_with_knowledge_and_best_now(action, env.best_action_record, action_trend,
                                                                env.best_action_choice, p_exp, env.all_last_action)
                
                
                action = utils.action_corrext_for_bps(env, action, p_rand, p_exp, action_trend)
                if hit_cnt > 0:
                    env.human_exp_hitcnt += 1
            print('action after shape:', action)
            
            # action_nor = action


            # buffer pool don't change over 1/2 or 2
            if env.info == 'CE':
                if env.last_action != -2:
                    bp_size = action_mapping(action[0], env.min_info[0], env.max_info[0])
                    # np.random.seed(time.time())
                    if bp_size > env.last_bp_size * 2:
                        # 如果当前预测值大于2倍上次值，则在[last_size,2*last_size]中随机一个值
                        # print("random.randint big")
                        bp_size = np.random.randint(env.last_bp_size, env.last_bp_size * 2)
                    elif bp_size < env.last_bp_size * 0.5:
                        # 如果当前预测值小于0.5倍上次值，则在[0.5*last_size,last_size]中随机一个值
                        # print("random.randint small")
                        bp_size = np.random.randint(env.last_bp_size * 0.5, env.last_bp_size)
                    # action_nor[0] = 
                    action[0] = utils.real_action_to_action(bp_size, env.min_info[0], env.max_info[0])
            elif env.info == 'NODES' and (not USE_LD_ADAPTOR):
                if env.last_action != -2:
                    # 依次修改node
                    index = 0
                    for se in env.se_info:
                        # buf_key = str(se.uuid) + '#buffer_pool_size'
                        bp_size_se = action_mapping(action[index], se.tune_action['buffer_pool_size'][1],
                                                    se.tune_action['buffer_pool_size'][2])
                        last_se_bp_size = se.last_bp_size
                        if bp_size_se > last_se_bp_size * 2:
                            # 如果当前预测值大于2倍上次值，则在[last_size,2*last_size]中随机一个值
                            # print("random.randint big")
                            bp_size_se = np.random.randint(last_se_bp_size, last_se_bp_size * 2)
                        elif bp_size_se < last_se_bp_size * 0.5:
                            # 如果当前预测值小于0.5倍上次值，则在[0.5*last_size,last_size]中随机一个值
                            # print("random.randint small")
                            bp_size_se = np.random.randint(last_se_bp_size * 0.5, last_se_bp_size)
                        action[index] = utils.real_action_to_action(bp_size_se,
                                                                    se.tune_action['buffer_pool_size'][1],
                                                                    se.tune_action['buffer_pool_size'][2])
                        index += len(se.tune_action)

                    for ce in env.ce_info:
                        # buf_key = ce.uuid + '#buffer_pool_size'
                        bp_size_ce = action_mapping(action[index], ce.tune_action['buffer_pool_size'][1],
                                                    ce.tune_action['buffer_pool_size'][2])
                        last_ce_bp_size = ce.last_bp_size
                        # ce
                        if bp_size_ce > last_ce_bp_size * 2:
                            # 如果当前预测值大于2倍上次值，则在[last_size,2*last_size]中随机一个值
                            # print("random.randint big")
                            bp_size_ce = np.random.randint(last_ce_bp_size, last_ce_bp_size * 2)
                        elif bp_size_ce < last_ce_bp_size * 0.5:
                            # 如果当前预测值小于0.5倍上次值，则在[0.5*last_size,last_size]中随机一个值
                            # print("random.randint small")
                            bp_size_ce = np.random.randint(last_ce_bp_size * 0.5, last_ce_bp_size)
                        action[index] = utils.real_action_to_action(bp_size_ce,
                                                                    ce.tune_action['buffer_pool_size'][1],
                                                                    ce.tune_action['buffer_pool_size'][2])
                        index += len(ce.tune_action)

            # print('action after filter: ', action)
            TD3_logger.info("\n[{}] Action: {}".format(env.method, action))
            # print("alpha = ", np.array(self.agent.alg.alpha))
            
            if env.state == 0 and USE_LHS:
                action = env.lhs_sampler.get_next_sample()
            
            next_obs, reward, done, info = env.step(action)
            if env.ld_adaptor is not None:
                # if USE_EXPERT_EXP:
                #     action_pred = env.ld_adaptor.reverse_transform(action)
                # else:
                #     action_pred = action_ld
                # print("\n[LD REVERSE]=====================================")
                # print("action = ", action)
                # print("action_pred = ", action_pred)
                # print("new_action = ", env.ld_adaptor.transform(action_pred))
                # print("=====================================[LD REVERSE]\n")
                
                env.all_last_action_2 = env.all_last_action
                env.all_last_action = action

            
            
            if done:
                print('+++ reset step +++')
                continue
            actual_obs = next_obs

            if env.state != 0 and TWO_PHASE == True and USE_STATUS_DR == True and not USE_FIXED_DR_KNOBS:
                next_obs = pca.transform(next_obs.reshape(1, -1))
                next_obs = np.array(next_obs).flatten()
            print("-----------------------------------")
            print("obs = ", obs)
            print("obs.type = ", type(obs))
            print("env.all_last_action = ", env.all_last_action)
            print("env.all_last_action_2 = ", env.all_last_action_2)
            print("-----------------------------------")
            cal_td_e_obs = agent.normalizer(obs, rpm)
            cal_td_e_next_obs = agent.normalizer(next_obs, rpm)
            if env.method != 'DQN':
                if USE_PRIORITY_RPM:
                    if USE_LD_ADAPTOR:
                        action = [action_pred]  # 方便存入replay memory
                    else:
                        action = [action]
                    obs_new = np.array([cal_td_e_obs]).astype('float32')
                    action_new = np.array([action]).astype('float32').reshape(1, env.action_dim)
                    reward_new = np.array([REWARD_SCALE * reward]).astype('float32')
                    next_obs_new = np.array([cal_td_e_next_obs]).astype('float32')
                    done_new = np.array([done]).astype('float32')
                    #print("action_new = ", action_new)
                    #print("obs_new = ", obs_new)
                    #print("reward_new = ", reward_new)
                    #print("next_obs_new = ", next_obs_new)
                    #print("done_new = ", done_new)
                    trans_reward_new = reward_new.copy()
                    trans_reward_new[trans_reward_new>0] = trans_reward_new[trans_reward_new>0]/1000000.0
                    
                    td_error = self.agent.cal_td_error(obs_new, action_new, trans_reward_new, next_obs_new, done_new)
                    #td_error = self.agent.cal_td_error(obs_new, action_new, reward_new, next_obs_new, done_new)
                    #td_error = self.agent.cal_td_error(obs_new, action_new, reward_new, next_obs_new, done_new)

                    rpm.append(td_error, (obs, action, REWARD_SCALE * reward, next_obs, done))
                else:
                    if USE_LD_ADAPTOR:
                        action = [action_pred]  # 方便存入replay memory
                    else:
                        action = [action_pred]                    
                    last_action = [env.all_last_action_2]
                    print("[MERGE DEBUG] rpm append last action shape = ", np.array(env.all_last_action_2).shape)
                    print("[TRANSFER DEBUG] obs = ", obs)
                    print("[TRANSFER DEBUG] action = ", action)
                    print("[TRANSFER DEBUG] reward = ", REWARD_SCALE * reward)
                    print("[TRANSFER DEBUG] next_obs = ", next_obs)
                    print("[TRANSFER DEBUG] done = ", done)
                    print("[TRANSFER DEBUG] last_action = ", last_action)
                    
                    # (batch_obs, batch_action, batch_reward, batch_next_obs, batch_done, batch_last_action) = rpm.buffer[0]
                    # print("[TRANSFER DEBUG] batch_obs = ", batch_obs)
                    # print("[TRANSFER DEBUG] batch_action = ", batch_action)
                    # print("[TRANSFER DEBUG] batch_reward = ", batch_reward)
                    # print("[TRANSFER DEBUG] batch_next_obs = ", batch_next_obs)
                    # print("[TRANSFER DEBUG] batch_done = ", batch_done)
                    # print("[TRANSFER DEBUG] batch_last_action = ", batch_last_action)
                    print("[DOUBLE NETWORK MERGE] rpm_double 0 len = ", len(rpm_double))
                    # rpm.append((obs, action, REWARD_SCALE * reward, next_obs, done, last_action))
                    rpm.append((obs, action_1, REWARD_SCALE * reward, next_obs, done, last_action))
                    print("[DOUBLE NETWORK MERGE] rpm_double 0.5 len = ", len(rpm_double))
                    
                    # print("[SHAP][OBS COLLECT] obs = ", obs)
                    # print("[SHAP][OBS COLLECT] rear_obs = ", rear_obs)
                    # print("[SHAP][OBS COLLECT] next_obs = ", next_obs)
                    # print("[SHAP][OBS COLLECT] actual_obs = ", actual_obs)
                    # rpm_full_real.append((rear_obs, action, REWARD_SCALE * reward, actual_obs, done, last_action))
                    
                    
                    if USE_DOUBLE_NETWORK:
                        action = [action_trend_2]
                        print("[DOUBLE NETWORK MERGE] rpm_double 1 len = ", len(rpm_double))
                        rpm_double.append((obs, action, REWARD_SCALE * reward, next_obs, done, last_action))
                        print("[DOUBLE NETWORK MERGE] rpm_double append exp [action] = ", action)
                        print("[DOUBLE NETWORK MERGE] rpm_double 2 len = ", len(rpm_double))
                    
                    
            else:
                rpm.append((obs, action, REWARD_SCALE * reward, next_obs, done))
                rpm.append_DQN_action(raw_action)
            # 平均得分
            mean_step_reward = float(env.score) / env.steps
            f_step_reward.write("scores=" + str(mean_step_reward) + "\n")
            f_step_reward.flush()

            if len(rpm) >= BATCH_SIZE:# and (steps % 5) == 0:
                if env.method != 'DQN':
                    #(batch_obs, batch_action, batch_reward, batch_next_obs,
                    #batch_done) = rpm.sample(BATCH_SIZE)
                    idxs = None
                    batch_obs = None
                    batch_action = None
                    batch_reward = None
                    batch_next_obs = None
                    batch_done = None
                    batch_last_action = None
                    if USE_PRIORITY_RPM:
                        (idxs, batch_obs, batch_action, batch_reward, batch_next_obs, batch_done) = rpm.sample(BATCH_SIZE)
                    else:
                        (batch_obs, batch_action, batch_reward, batch_next_obs, batch_done, batch_last_action) = rpm.sample(BATCH_SIZE)

                else:
                    (batch_obs, batch_action, batch_reward, batch_next_obs,
                    batch_done) = rpm.DQN_sample(BATCH_SIZE)
                # 这里维度已经缩小了，所以不再需要pca降维，降维应该发生在收集到原始数据库状态后，但是需要进行数据标准化！！
                batch_obs = agent.normalizerBatch(batch_obs, rpm)
                batch_next_obs = agent.normalizerBatch(batch_next_obs, rpm)
                # DDPG
                if env.method == 'DDPG':
                    critic_cost = agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)
                # TD3
                if env.method == 'TD3' or env.method == 'SAC' or env.method == 'SAC_2':
                    #if USE_PRIORITY_RPM:
                        #new_td_error = self.agent.cal_td_error(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)
                        #print("*************************************")
                        #print("*new_td_error_before_learn = {0}".format(new_td_error))
                        #print("*************************************")
                    
                    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    print("agent learn, alg = {0}, time = run_epsisode".format(env.method))
                    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

                    # print(batch_action)
                    
                    resize_batch_reward = batch_reward.copy()
                    critic_cost = 0
                    actor_cost, critic_cost = agent.learn(batch_obs, batch_action, resize_batch_reward, batch_next_obs,
                                                        batch_done, batch_last_action)
                    
                    
                    
                    if USE_DOUBLE_NETWORK:
                        batch_action_double = []
                        print("[DOUBLE NETWORK MERGE] batch_action.shape = ", batch_action.shape)
                        
                        # for i in range(batch_action.shape[0]):
                        #     print("[DOUBLE LEARN DEBUG] batch_obs[i].shape = {0}".format(batch_obs[i].shape))
                        #     print("[DOUBLE LEARN DEBUG] batch_last_action[i].shape = {0}".format(batch_last_action[i].shape))
                            
                        #     action_double = self.agent_double.get_trend(batch_obs[i], np.array([batch_last_action[i]]))
                        #     batch_action_double.append(action_double)
                        # batch_action_double = np.array(batch_action_double)
                        
                        batch_idxs = []
                        hash_obs = {}
                        for i in range(len(rpm_double.buffer)):
                            # print("[DOUBLE LEARN DEBUG] put obs = {0}".format(np.float32(rpm_double.buffer[i][2])))
                            hash_obs[np.float32(rpm_double.buffer[i][2])] = i
                        for i in range(batch_action.shape[0]):
                            selected_obs = batch_reward[i]
                            # print("[DOUBLE LEARN DEBUG] selected_obs = {0}".format(selected_obs))
                            batch_idxs.append(hash_obs[selected_obs])
                            # print("[DOUBLE LEARN DEBUG] batch_idxs = {0}".format(batch_idxs))
                        print("[DOUBLE LEARN DEBUG] batch_idxs = {0}".format(batch_idxs))
                        for idx in batch_idxs:
                            batch_action_double.append(np.array(rpm_double.buffer[idx][1][0]).astype('float32'))
                        print("[DOUBLE LEARN DEBUG] batch_action_double = {0}".format(batch_action_double))
                        
                            
                            
                            
                        batch_action_double = np.array(batch_action_double)
                        
                        print("[DOUBLE NETWORK MERGE] batch_action_double.shape = ", batch_action_double.shape)
                        actor_cost_double, critic_cost_double = self.agent_double.learn(batch_obs, batch_action_double, batch_reward, batch_next_obs,
                                                        batch_done, batch_last_action)
                    

                    
                    
                    if USE_PRIORITY_RPM:
                        trans_batch_reward = batch_reward.copy()
                        trans_batch_reward[trans_batch_reward>0] = trans_batch_reward[trans_batch_reward>0] / 1000000.0
                        new_td_error = self.agent.cal_td_error(batch_obs, batch_action, trans_batch_reward, batch_next_obs, batch_done)
                        #new_td_error = self.agent.cal_td_error(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)
                        print("*************************************")
                        print("*new_td_error = {0}, critic_cost = {1}".format(new_td_error, critic_cost))
                        print("*************************************")
                    #sum_error = 0
                    #for i in range(BATCH_SIZE):
                        #print("+++++ obs = ", batch_obs)
                        #print("+++++ action = ", batch_action)
                        #print("+++++ reward = ", batch_reward)
                        #print("+++++ next_obs = ", batch_next_obs)
                        #print("+++++ done = ", batch_done)
                        #obs_new = np.array([batch_obs[i]]).astype('float32')
                        #action_new = np.array([batch_action[i]]).astype('float32').reshape(1, env.action_dim)
                        #reward_new = np.array([batch_reward[i]]).astype('float32')
                        #next_obs_new = np.array([batch_next_obs[i]]).astype('float32')
                        #done_new = np.array([batch_done[i]]).astype('float32')
                
                        #tmp_td_error = self.agent.cal_td_error(obs_new, action_new, reward_new, next_obs_new, done_new)
                        #sum_error += tmp_td_error
                        #print("=================== single td error = ", tmp_td_error)
                    #print("********************** avg td error = ", sum_error / BATCH_SIZE)
                        for i in range(BATCH_SIZE):
                            idx = idxs[i]
                            rpm.update(idx, new_td_error[i])                

                # DQN
                if env.method == 'DQN':
                    critic_cost = agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)

                if env.state != 0:
                    # print("1111111111111111111112222222222222222222222", critic_cost)
                    if env.method == 'SAC' or env.method == 'SAC_2':
                        accumulate_loss += critic_cost
                    else:
                        accumulate_loss += critic_cost[0]
                    critic_cost_mean = accumulate_loss / env.steps
                    with open("./test_model/critic_loss.txt", "a+", encoding="utf-8") as loss_f:
                        loss_f.write("critic_loss="+str(critic_cost_mean)+"\n")

            if env.info == 'CE':
                TD3_logger.info(
                    "\n[{}][Episode: {}][Step: {}][Metric qps:{} hit_ratio:{} buffer_size:{}]Reward: {} Score: {} Done: {}".format(
                        env.method, env.episode, steps, env.last_qps, env.last_hr, env.last_bp_size, reward, total_reward, done
                    ))
            elif env.info == 'NODES':
                s = '[' + env.method + '][Episode: ' + str(env.episode) + '][Step: ' + str(steps) + ']'
                s += '[ ' + str(env.se_num) + ' ses:'
                for se in env.se_info:
                    s += '{se' + str(se.uuid) + ', hit_ratio: ' + str(se.last_hr) + ', buffer_pool_size: ' + str(se.last_bp_size) + '}'
                s += '][ ' + str(env.ce_num) + ' ces:'
                for ce in env.ce_info:
                    if ce.is_primary == True:
                        s += '{ce' + str(ce.uuid) + ', qps: ' + str(ce.last_qps) + ', hit_ratio: ' + str(
                            ce.last_hr) + ', buffer_pool_size: ' + str(ce.last_bp_size) + '}'
                    else:
                        s += '{ce' + str(ce.uuid) + ', hit_ratio: ' + str(
                            ce.last_hr) + ', buffer_pool_size: ' + str(ce.last_bp_size) + '}'
                s += ']'
                s += 'Reward: ' + str(reward) + ', Score: ' + str(total_reward) + ', Done: ' + str(done)
                TD3_logger.info("\n{}".format(s))

            # if steps % 10 == 0:
            #     # 保存模型
            #     ckpt = './test_model/test_save/test_save_{}'.format(int(time.time()))
            #     # f = open("./test_model/test_save/ce_rpm.txt", "wb")
            #     print('-----------save_model-----------')
            #     print('ckpt = ', ckpt)
            #
            #     # DDPG
            #     if env.method == 'DDPG':
            #         agent.save(save_path=ckpt+'.cpkt')
            #     # TD3
            #     if env.method == 'TD3':
            #         agent.save(save_path=ckpt+'_predict.ckpt', mode='predict')
            #         agent.save(save_path=ckpt+'_train_actor.ckpt', mode='train_actor')
            #         agent.save(save_path=ckpt+'_train_critic.ckpt', mode='train_critic')

            obs = actual_obs
            total_reward += reward
            if env.state == 0:
                if steps >= WARMUP_MOVE_STEPS or total_reward < -50 or total_reward > 20000000:
                    print('WARMUP DONE : steps = ', steps)
                    break
            else:
                if steps >= MOVE_STEPS or done or total_reward < -50 or total_reward > 20000000:
                    print('DONE : steps = ', steps)
                    break
        env.end_time = time.time()
        return total_reward, steps



    def grid_search_for_seed(self, seed, act_dim, obs_dim, exp_gen_rpm_train, exp_gen_rpm_test, env):
        # return 0.00
        max_action = 0.99999999
        ld_adaptor = LowDimAdaptor(env, globalValue.HES_LOW_ACT_DIM, seed)
        
        
        # 动态命名资源，例如模型
        variable_name = f"model_seed_{seed}"  # 每个进程有独立的模型名称
        resources = {}
        algs = {}
        agents = {}
        
        # 动态创建模型对象并存储在字典中
        resources[variable_name] =  SACModel(act_dim, obs_dim)
        algs[variable_name] = SAC_2(
            actor=resources[variable_name].actor_model,
            critic=resources[variable_name].critic_model,
            alpha_model=resources[variable_name].alpha_model,
            reward_model=resources[variable_name].reward_model,
            max_action=max_action,
            alpha=H_SAC_ALPHA,
            gamma=H_SAC_GAMMA,
            tau=H_SAC_TAU,
            actor_lr=H_SAC_ACTOR_LR,
            critic_lr=H_SAC_CRITIC_LR,
            alpha_lr=H_SAC_ALPHA_LR,
            batch_size=BATCH_SIZE,
            states_model=resources[variable_name].states_model,
            
        )
        
        ld = act_dim
        hd = act_dim
        if USE_LD_ADAPTOR:
            ld = ld_adaptor.target_dim
            hd = ld_adaptor.high_dim
        
        agents[variable_name] = SAC2Agent(algs[variable_name], obs_dim, act_dim, ld, hd)
        # agents[variable_name].build_program() 
        
        
        total_loss = 0
        loss_1 = 0
        loss_2 = 0
        test_continue_num = 3
        delta_step = 200
        train_episode = delta_step+1
        learn_t = 1
        for i in range(train_episode):
            # for j in range(EXP_GEN_LEARN_BATCH_SIZE):
            (exp_batch_obs, exp_batch_action, exp_batch_reward, exp_batch_next_obs, exp_batch_done, exp_batch_last_action, exp_batch_trend) = \
                exp_gen_rpm_train.exp_fast_learn_sample(EXP_GEN_LEARN_BATCH_SIZE * learn_t)
        
            # n_exp_batch_obs = exp_batch_obs 
            # n_exp_batch_next_obs = exp_batch_next_obs 
            n_exp_batch_obs = agents[variable_name].normalizerBatch(exp_batch_obs, exp_gen_rpm_train)
            n_exp_batch_next_obs = agents[variable_name].normalizerBatch(exp_batch_next_obs, exp_gen_rpm_train)
            # print("[MERGE DEBUG] exp_batch_trend.shape = ", exp_batch_trend.shape)
            # print("[MERGE DEBUG] exp_batch_last_action.shape = ", exp_batch_last_action.shape)
            # print("[MERGE DEBUG] exp_batch_trend.shape = ", exp_batch_trend.shape)
            A = []
            if USE_LD_ADAPTOR:
                A = np.stack([ld_adaptor.A] * EXP_GEN_LEARN_BATCH_SIZE * learn_t, axis=0).astype(np.float32)
            else:
                # A = np.eye(env.action_dim)
                A = np.stack([np.eye(act_dim)] * EXP_GEN_LEARN_BATCH_SIZE * learn_t, axis=0).astype(np.float32)
            # print("[MERGE DEBUG] A shape = ", A.shape)
            
            # print("[GRID SEARCH] n_exp_batch_obs.shape = ", n_exp_batch_obs.shape)
            # print("[GRID SEARCH] exp_batch_last_action.shape = ", exp_batch_last_action.shape)
            # print("[GRID SEARCH] exp_batch_trend.shape = ", exp_batch_trend.shape)
            # print("[GRID SEARCH] A.shape = ", A.shape)
            
            
            loss = agents[variable_name].exp_fast_learn(n_exp_batch_obs, exp_batch_last_action, exp_batch_trend, A)
            # total_loss += loss
            
            total_loss += loss
            if i % (delta_step/10) == 0:
                print("[FAST LEARN {0}][SEED={1}] loss = {2}".format(i, seed, total_loss * 1.0 / (i+1)))

            # 测试
            
            if i % delta_step == 0 and i != 0:
                (exp_batch_obs, exp_batch_action, exp_batch_reward, exp_batch_next_obs, exp_batch_done, exp_batch_last_action, exp_batch_trend) = \
                    exp_gen_rpm_test.exp_fast_learn_sample(EXP_GEN_LEARN_BATCH_SIZE)
                # n_exp_batch_obs = exp_batch_obs
                n_exp_batch_obs = agents[variable_name].normalizerBatch(exp_batch_obs, exp_gen_rpm_test)
                
                sum_hit_cnt = 0
                total_hit_ratio_single = 0.0
                for j in range(EXP_GEN_LEARN_BATCH_SIZE):
                    
                    pred_action = agents[variable_name].get_trend(n_exp_batch_obs[j], np.array([exp_batch_last_action[j]]).astype('float32'))
                    last_action = exp_batch_last_action[j]
                    exp_trend = exp_batch_trend[j]
                    exp_trend = exp_trend - 1.0
                    pred_action = pred_action - 1.0
                    
                    
                    
                    
                    if USE_LD_ADAPTOR:
                        pred_action = ld_adaptor.transform(pred_action)
                    
                    # print("[FL DUBEG] pred_action = ", (list)(pred_action))
                    # print("[FL DUBEG] exp_trend = ", (list)(exp_trend))
                    
                    # delta_action = pred_action - last_action
                    delta_action = pred_action
                        
                    sign_delta_action = np.sign(delta_action)
                    product_sign = np.multiply(sign_delta_action, exp_trend)
                    product_sign = np.sign(product_sign)                        
                    # hit_cnt = np.sum(product_sign, axis=0)
                    # if hit_cnt > 0:
                    #     sum_hit_cnt += 1
                    hit_cnt = np.sum(product_sign, axis=0)
                    abs_cnt = np.sum(np.abs(product_sign), axis=0)
                    hit_r_single = hit_cnt * 1.0 / abs_cnt
                    # print("[FL DUBEG] hit_r_single = ", hit_r_single)
                    total_hit_ratio_single += hit_r_single
                    
                    if hit_r_single > EXP_GEN_HIT_THRES:
                        sum_hit_cnt += 1
                    
                test_rate = sum_hit_cnt / EXP_GEN_LEARN_BATCH_SIZE
                print("[FL DUBEG] total_hit_ratio_single = ", total_hit_ratio_single)
                print("[FAST LEARN {0}] evaluate USE_EXP_GEN_FAST_LERAN use gen test, rate = {1}, rate_single = {2}".format(i, sum_hit_cnt / EXP_GEN_LEARN_BATCH_SIZE, total_hit_ratio_single / EXP_GEN_LEARN_BATCH_SIZE))
                
        return total_hit_ratio_single / EXP_GEN_LEARN_BATCH_SIZE
      
    
    def grid_search_for_seed_without_data(self, seed):
        self.env = env = NodesEnv()
        env.init_nodes()
        act_dim = globalValue.HES_LOW_ACT_DIM
        obs_dim = env.state_dim
        
        
        
        # return 0.00
        max_action = 0.99999999
        env.ld_adaptor = ld_adaptor = LowDimAdaptor(env, globalValue.HES_LOW_ACT_DIM, seed)
        env.action_dim = env.ld_adaptor.target_dim
        
        # 动态命名资源，例如模型
        variable_name = f"model_seed_{seed}"  # 每个进程有独立的模型名称
        resources = {}
        algs = {}
        agents = {}
        
        # 动态创建模型对象并存储在字典中
        resources[variable_name] =  SACModel(act_dim, obs_dim)
        algs[variable_name] = SAC_2(
            actor=resources[variable_name].actor_model,
            critic=resources[variable_name].critic_model,
            alpha_model=resources[variable_name].alpha_model,
            reward_model=resources[variable_name].reward_model,
            max_action=max_action,
            alpha=H_SAC_ALPHA,
            gamma=H_SAC_GAMMA,
            tau=H_SAC_TAU,
            actor_lr=H_SAC_ACTOR_LR,
            critic_lr=H_SAC_CRITIC_LR,
            alpha_lr=H_SAC_ALPHA_LR,
            batch_size=BATCH_SIZE,
            states_model=resources[variable_name].states_model,
            
        )
        
        ld = act_dim
        hd = act_dim
        if USE_LD_ADAPTOR:
            ld = ld_adaptor.target_dim
            hd = ld_adaptor.high_dim
        
        agents[variable_name] = SAC2Agent(algs[variable_name], obs_dim, act_dim, ld, hd)
        # agents[variable_name].build_program() 
        
        
        if USE_EXP_GEN_FAST_LERAN:
            env.lerc = LERC(obs_dim, act_dim, True, BATCH_SIZE, len(globalValue.CONNECT_CE_IP) + len(globalValue.CONNECT_SE_IP), USE_LD_ADAPTOR, env.ld_adaptor)
            # 生成专家模拟经验
            exp_s_list, exp_a_list, exp_r_list, exp_n_s_list, exp_d_list, exp_l_a_list, exp_trend_list = \
                env.lerc.generate_expert_exp(EXP_GEN_TOTAL_NUM, env)
            
            # print("GENERTATE {0} EXPERT EXP".format(EXP_GEN_TOTAL_NUM))
            # print("exp_s_list shape = ", exp_s_list.shape)
            # print("exp_a_list shape = ", exp_a_list.shape)
            # print("exp_r_list shape = ", exp_r_list.shape)
            # print("exp_n_s_list shape = ", exp_n_s_list.shape)
            # print("exp_d_list shape = ", exp_d_list.shape)
            # print("exp_l_a_list shape = ", exp_l_a_list.shape)
            # print("exp_trend_list shape = ", exp_trend_list.shape)
            
            # rpm_act_dim = act_dim
            # if USE_LD_ADAPTOR
            ld = act_dim
            hd = act_dim
            if USE_LD_ADAPTOR:
                ld = env.ld_adaptor.target_dim
                hd = env.ld_adaptor.high_dim
            exp_gen_rpm_train = ReplayMemory(MEMORY_SIZE, act_dim, obs_dim, USE_LD_ADAPTOR, ld, hd)
            exp_gen_rpm_test = ReplayMemory(MEMORY_SIZE, act_dim, obs_dim, USE_LD_ADAPTOR, ld, hd)
            for i in range(int(EXP_GEN_TOTAL_NUM * 0.8)):
                exp_gen_rpm_train.append((exp_s_list[i], exp_a_list[i], exp_r_list[i], exp_n_s_list[i], exp_d_list[i], exp_l_a_list[i], exp_trend_list[i]))
            for i in range(int(EXP_GEN_TOTAL_NUM * 0.8), EXP_GEN_TOTAL_NUM):
                exp_gen_rpm_test.append((exp_s_list[i], exp_a_list[i], exp_r_list[i], exp_n_s_list[i], exp_d_list[i], exp_l_a_list[i], exp_trend_list[i]))
                
                
            # if EXP_GEN_USE_ACTUAL_ENV_TEST:
            #     exp_gen_rpm_actual_env_file = open(EXP_GEN_ACTUAL_TEST_FILE, "rb")
            #     exp_gen_rpm_actual_env = pickle.load(exp_gen_rpm_actual_env_file)
            
            
            
        total_loss = 0
        loss_1 = 0
        loss_2 = 0
        test_continue_num = 3
        delta_step = 200
        train_episode = delta_step+1
        learn_t = 1
        for i in range(train_episode):
            # for j in range(EXP_GEN_LEARN_BATCH_SIZE):
            (exp_batch_obs, exp_batch_action, exp_batch_reward, exp_batch_next_obs, exp_batch_done, exp_batch_last_action, exp_batch_trend) = \
                exp_gen_rpm_train.exp_fast_learn_sample(EXP_GEN_LEARN_BATCH_SIZE * learn_t)
        
            # n_exp_batch_obs = exp_batch_obs 
            # n_exp_batch_next_obs = exp_batch_next_obs 
            n_exp_batch_obs = agents[variable_name].normalizerBatch(exp_batch_obs, exp_gen_rpm_train)
            n_exp_batch_next_obs = agents[variable_name].normalizerBatch(exp_batch_next_obs, exp_gen_rpm_train)
            # print("[MERGE DEBUG] exp_batch_trend.shape = ", exp_batch_trend.shape)
            # print("[MERGE DEBUG] exp_batch_last_action.shape = ", exp_batch_last_action.shape)
            # print("[MERGE DEBUG] exp_batch_trend.shape = ", exp_batch_trend.shape)
            A = []
            if USE_LD_ADAPTOR:
                A = np.stack([ld_adaptor.A] * EXP_GEN_LEARN_BATCH_SIZE * learn_t, axis=0).astype(np.float32)
            else:
                # A = np.eye(env.action_dim)
                A = np.stack([np.eye(act_dim)] * EXP_GEN_LEARN_BATCH_SIZE * learn_t, axis=0).astype(np.float32)
            # print("[MERGE DEBUG] A shape = ", A.shape)
            
            # print("[GRID SEARCH] n_exp_batch_obs.shape = ", n_exp_batch_obs.shape)
            # print("[GRID SEARCH] exp_batch_last_action.shape = ", exp_batch_last_action.shape)
            # print("[GRID SEARCH] exp_batch_trend.shape = ", exp_batch_trend.shape)
            # print("[GRID SEARCH] A.shape = ", A.shape)
            
            
            loss = agents[variable_name].exp_fast_learn(n_exp_batch_obs, exp_batch_last_action, exp_batch_trend, A)
            # total_loss += loss
            
            total_loss += loss
            if i % (delta_step/10) == 0:
                print("[FAST LEARN {0}][SEED={1}] loss = {2}".format(i, seed, total_loss * 1.0 / (i+1)))

            # 测试
            
            if i % delta_step == 0 and i != 0:
                (exp_batch_obs, exp_batch_action, exp_batch_reward, exp_batch_next_obs, exp_batch_done, exp_batch_last_action, exp_batch_trend) = \
                    exp_gen_rpm_test.exp_fast_learn_sample(EXP_GEN_LEARN_BATCH_SIZE)
                # n_exp_batch_obs = exp_batch_obs
                n_exp_batch_obs = agents[variable_name].normalizerBatch(exp_batch_obs, exp_gen_rpm_test)
                
                sum_hit_cnt = 0
                total_hit_ratio_single = 0.0
                for j in range(EXP_GEN_LEARN_BATCH_SIZE):
                    
                    pred_action = agents[variable_name].get_trend(n_exp_batch_obs[j], np.array([exp_batch_last_action[j]]).astype('float32'))
                    last_action = exp_batch_last_action[j]
                    exp_trend = exp_batch_trend[j]
                    exp_trend = exp_trend - 1.0
                    pred_action = pred_action - 1.0
                    
                    
                    
                    
                    if USE_LD_ADAPTOR:
                        pred_action = ld_adaptor.transform(pred_action)
                    
                    # print("[FL DUBEG] pred_action = ", (list)(pred_action))
                    # print("[FL DUBEG] exp_trend = ", (list)(exp_trend))
                    
                    # delta_action = pred_action - last_action
                    delta_action = pred_action
                        
                    sign_delta_action = np.sign(delta_action)
                    product_sign = np.multiply(sign_delta_action, exp_trend)
                    product_sign = np.sign(product_sign)                        
                    # hit_cnt = np.sum(product_sign, axis=0)
                    # if hit_cnt > 0:
                    #     sum_hit_cnt += 1
                    hit_cnt = np.sum(product_sign, axis=0)
                    abs_cnt = np.sum(np.abs(product_sign), axis=0)
                    hit_r_single = hit_cnt * 1.0 / abs_cnt
                    # print("[FL DUBEG] hit_r_single = ", hit_r_single)
                    total_hit_ratio_single += hit_r_single
                    
                    if hit_r_single > EXP_GEN_HIT_THRES:
                        sum_hit_cnt += 1
                    
                test_rate = sum_hit_cnt / EXP_GEN_LEARN_BATCH_SIZE
                print("[FL DUBEG] total_hit_ratio_single = ", total_hit_ratio_single)
                print("[FAST LEARN {0}][SEED={1}] evaluate USE_EXP_GEN_FAST_LERAN use gen test, rate = {2}, rate_single = {3}".format(i, seed, sum_hit_cnt / EXP_GEN_LEARN_BATCH_SIZE, total_hit_ratio_single / EXP_GEN_LEARN_BATCH_SIZE))
                
        return total_hit_ratio_single / EXP_GEN_LEARN_BATCH_SIZE
      
      
    def grid_search_for_seed_without_data_for_double(self, pre_seed, seed):
        self.env = env = NodesEnv()
        env.init_nodes()
        act_dim = globalValue.HES_LOW_ACT_DIM
        obs_dim = env.state_dim
        
        
        self.hes_low_seed = pre_seed
        
        # return 0.00
        max_action = 0.99999999
        env.ld_adaptor = ld_adaptor = LowDimAdaptor(env, globalValue.HES_LOW_ACT_DIM, self.hes_low_seed)
        env.ld_adaptor_double = ld_adaptor_double = LowDimAdaptor(env, globalValue.HES_LOW_ACT_DIM, seed)
        env.ld_adaptor_double.double_network_correct(env.ld_adaptor.A)
        env.ld_adaptor_double.double_network_correct(env.ld_adaptor.A)
        
        if seed == 40:
            print("[GRID SERACH DEBUG] pre_seed = ", self.hes_low_seed)
            print("[GRID SERACH DEBUG] ld_adaptor.A = ", ld_adaptor.A)
            print("[GRID SERACH DEBUG] ld_adaptor_double.A = ", ld_adaptor_double.A)
            
            
        # time.sleep(10000)
        
        env.action_dim = env.ld_adaptor.target_dim
        
        # 动态命名资源，例如模型
        variable_name = f"model_seed_{seed}"  # 每个进程有独立的模型名称
        resources = {}
        resources_double = {}
        algs = {}
        algs_double = {}
        agents = {}
        agents_double = {}
        
        # 动态创建模型对象并存储在字典中
        resources[variable_name] =  SACModel(act_dim, obs_dim)
        algs[variable_name] = SAC_2(
            actor=resources[variable_name].actor_model,
            critic=resources[variable_name].critic_model,
            alpha_model=resources[variable_name].alpha_model,
            reward_model=resources[variable_name].reward_model,
            max_action=max_action,
            alpha=H_SAC_ALPHA,
            gamma=H_SAC_GAMMA,
            tau=H_SAC_TAU,
            actor_lr=H_SAC_ACTOR_LR,
            critic_lr=H_SAC_CRITIC_LR,
            alpha_lr=H_SAC_ALPHA_LR,
            batch_size=BATCH_SIZE,
            states_model=resources[variable_name].states_model,
            
        )
        
        resources_double[variable_name] =  SACModel(env.ld_adaptor_double.target_dim, self.env.state_dim, True)
        algs_double[variable_name] = SAC_2(
            actor=resources_double[variable_name].actor_model,
            critic=resources_double[variable_name].critic_model,
            alpha_model=resources_double[variable_name].alpha_model,
            reward_model=resources_double[variable_name].reward_model,
            max_action=max_action,
            alpha=H_SAC_ALPHA,
            gamma=H_SAC_GAMMA,
            tau=H_SAC_TAU,
            actor_lr=H_SAC_ACTOR_LR,
            critic_lr=H_SAC_CRITIC_LR,
            alpha_lr=H_SAC_ALPHA_LR,
            batch_size=BATCH_SIZE,
            states_model=resources_double[variable_name].states_model,
            is_double=True,
        )
        
        ld = act_dim
        hd = act_dim
        if USE_LD_ADAPTOR:
            ld = ld_adaptor.target_dim
            hd = ld_adaptor.high_dim
        
        agents[variable_name] = SAC2Agent(algs[variable_name], obs_dim, act_dim, ld, hd)
        agents_double[variable_name] = SAC2Agent(algs_double[variable_name], obs_dim, env.ld_adaptor_double.target_dim, env.ld_adaptor_double.target_dim, hd)
        # agents[variable_name].build_program() 
        
        
        if USE_EXP_GEN_FAST_LERAN:
            env.lerc = LERC(obs_dim, act_dim, True, BATCH_SIZE, len(globalValue.CONNECT_CE_IP) + len(globalValue.CONNECT_SE_IP), USE_LD_ADAPTOR, env.ld_adaptor)
            # 生成专家模拟经验
            exp_s_list, exp_a_list, exp_r_list, exp_n_s_list, exp_d_list, exp_l_a_list, exp_trend_list = \
                env.lerc.generate_expert_exp(EXP_GEN_TOTAL_NUM, env)
            
            # print("GENERTATE {0} EXPERT EXP".format(EXP_GEN_TOTAL_NUM))
            # print("exp_s_list shape = ", exp_s_list.shape)
            # print("exp_a_list shape = ", exp_a_list.shape)
            # print("exp_r_list shape = ", exp_r_list.shape)
            # print("exp_n_s_list shape = ", exp_n_s_list.shape)
            # print("exp_d_list shape = ", exp_d_list.shape)
            # print("exp_l_a_list shape = ", exp_l_a_list.shape)
            # print("exp_trend_list shape = ", exp_trend_list.shape)
            
            # rpm_act_dim = act_dim
            # if USE_LD_ADAPTOR
            ld = act_dim
            hd = act_dim
            if USE_LD_ADAPTOR:
                ld = env.ld_adaptor.target_dim
                hd = env.ld_adaptor.high_dim
            exp_gen_rpm_train = ReplayMemory(MEMORY_SIZE, act_dim, obs_dim, USE_LD_ADAPTOR, ld, hd)
            exp_gen_rpm_test = ReplayMemory(MEMORY_SIZE, act_dim, obs_dim, USE_LD_ADAPTOR, ld, hd)
            for i in range(int(EXP_GEN_TOTAL_NUM * 0.8)):
                exp_gen_rpm_train.append((exp_s_list[i], exp_a_list[i], exp_r_list[i], exp_n_s_list[i], exp_d_list[i], exp_l_a_list[i], exp_trend_list[i]))
            for i in range(int(EXP_GEN_TOTAL_NUM * 0.8), EXP_GEN_TOTAL_NUM):
                exp_gen_rpm_test.append((exp_s_list[i], exp_a_list[i], exp_r_list[i], exp_n_s_list[i], exp_d_list[i], exp_l_a_list[i], exp_trend_list[i]))
                
                
            if EXP_GEN_USE_ACTUAL_ENV_TEST:
                exp_gen_rpm_actual_env_file = open(EXP_GEN_ACTUAL_TEST_FILE, "rb")
                exp_gen_rpm_actual_env = pickle.load(exp_gen_rpm_actual_env_file)
            
            
            
        total_loss = 0
        loss_1 = 0
        loss_2 = 0
        test_continue_num = 3
        delta_step = 200
        train_episode = delta_step+1
        learn_t = 1
        for i in range(train_episode):
            # for j in range(EXP_GEN_LEARN_BATCH_SIZE):
            (exp_batch_obs, exp_batch_action, exp_batch_reward, exp_batch_next_obs, exp_batch_done, exp_batch_last_action, exp_batch_trend) = \
                exp_gen_rpm_train.exp_fast_learn_sample(EXP_GEN_LEARN_BATCH_SIZE * learn_t)
        
            # n_exp_batch_obs = exp_batch_obs 
            # n_exp_batch_next_obs = exp_batch_next_obs 
            n_exp_batch_obs = agents[variable_name].normalizerBatch(exp_batch_obs, exp_gen_rpm_train)
            n_exp_batch_next_obs = agents[variable_name].normalizerBatch(exp_batch_next_obs, exp_gen_rpm_train)
            # print("[MERGE DEBUG] exp_batch_trend.shape = ", exp_batch_trend.shape)
            # print("[MERGE DEBUG] exp_batch_last_action.shape = ", exp_batch_last_action.shape)
            # print("[MERGE DEBUG] exp_batch_trend.shape = ", exp_batch_trend.shape)
            A = []
            if USE_LD_ADAPTOR:
                A = np.stack([ld_adaptor.A] * EXP_GEN_LEARN_BATCH_SIZE * learn_t, axis=0).astype(np.float32)
            else:
                # A = np.eye(env.action_dim)
                A = np.stack([np.eye(act_dim)] * EXP_GEN_LEARN_BATCH_SIZE * learn_t, axis=0).astype(np.float32)
            # print("[MERGE DEBUG] A shape = ", A.shape)
            
            # print("[GRID SEARCH] n_exp_batch_obs.shape = ", n_exp_batch_obs.shape)
            # print("[GRID SEARCH] exp_batch_last_action.shape = ", exp_batch_last_action.shape)
            # print("[GRID SEARCH] exp_batch_trend.shape = ", exp_batch_trend.shape)
            # print("[GRID SEARCH] A.shape = ", A.shape)
            
            
            # loss = agents[variable_name].exp_fast_learn(n_exp_batch_obs, exp_batch_last_action, exp_batch_trend, A)
            # total_loss += loss
            A_1 = A
            A_2 = []
            A_2 = np.stack([ld_adaptor_double.A] * EXP_GEN_LEARN_BATCH_SIZE * learn_t, axis=0).astype(np.float32)
            exp_batch_trend_normal = exp_batch_trend - 1
            # print("[DOUBLE NETWORK DEBUG] exp_batch_trend_normal shape = ", exp_batch_trend_normal.shape)
            # print("[DOUBLE NETWORK DEBUG] exp_batch_trend_normal = ", exp_batch_trend_normal)
            batch_net_1_trend = agents[variable_name].get_trend(n_exp_batch_obs, np.array([exp_batch_last_action]).astype('float32')) - 1
            batch_net_2_trend = agents_double[variable_name].get_trend(n_exp_batch_obs, np.array([exp_batch_last_action]).astype('float32')) - 1
            # print("[DOUBLE NETWORK DEBUG] batch_net_1_trend shape = ", batch_net_1_trend.shape)
            # print("[DOUBLE NETWORK DEBUG] batch_net_1_trend = ", batch_net_1_trend)
            # print("[DOUBLE NETWORK DEBUG] batch_net_2_trend shape = ", batch_net_2_trend.shape)
            # print("[DOUBLE NETWORK DEBUG] batch_net_2_trend = ", batch_net_2_trend)
            batch_net_1_trend_hd = ld_adaptor.batch_transform(batch_net_1_trend)
            batch_net_2_trend_hd = ld_adaptor_double.batch_transform(batch_net_2_trend)
            # print("[DOUBLE NETWORK DEBUG] batch_net_1_trend_hd shape = ", batch_net_1_trend_hd.shape)
            # print("[DOUBLE NETWORK DEBUG] batch_net_2_trend_hd shape = ", batch_net_2_trend_hd.shape)
            # print("[DOUBLE NETWORK DEBUG] batch_net_1_trend_hd = ", batch_net_1_trend_hd)
            # print("[DOUBLE NETWORK DEBUG] batch_net_2_trend_hd = ", batch_net_2_trend_hd)
            batch_target_trend_1 = 2 * exp_batch_trend_normal - batch_net_2_trend_hd
            batch_target_trend_2_minus = (exp_batch_trend_normal - globalValue.DOUBLE_RATIO * batch_net_1_trend_hd) / (1-globalValue.DOUBLE_RATIO)
            batch_target_trend_2_sign = np.sign(batch_target_trend_2_minus)
            batch_target_trend_2 = (np.abs(batch_target_trend_2_minus) + 0.2) * np.abs(np.sign(exp_batch_trend_normal)) * batch_target_trend_2_sign
            # batch_target_trend_2 = batch_target_trend_2_minus
            # print("[DOUBLE NETWORK DEBUG] batch_target_trend_1 shape = ", batch_target_trend_1.shape)
            # print("[DOUBLE NETWORK DEBUG] batch_target_trend_2 shape = ", batch_target_trend_2.shape)
            # print("[DOUBLE NETWORK DEBUG] batch_target_trend_1 = ", batch_target_trend_1)
            # print("[DOUBLE NETWORK DEBUG] batch_target_trend_2 = ", batch_target_trend_2)
            batch_target_label_1 = np.sign(batch_target_trend_1) + 1
            # batch_target_label_2 = np.sign(batch_target_trend_2) + 1
            batch_target_label_2 = batch_target_trend_2
                    
                    
            loss = agents[variable_name].exp_fast_learn(n_exp_batch_obs, exp_batch_last_action, exp_batch_trend, A_1)
            loss_double = agents_double[variable_name].exp_fast_learn_double(n_exp_batch_obs, exp_batch_last_action, batch_target_label_2, A_2)
                
            
            loss_1 += loss
            loss_2 += loss_double
            total_loss += loss * globalValue.DOUBLE_RATIO + loss_double * (1-globalValue.DOUBLE_RATIO)
            if i % (delta_step/10) == 0:
                print("[FAST LEARN {0}][PRE_SEED={1}][SEED={2}] loss_1 = {3}, loss_2 = {4}, total_loss = {5}".format(i, self.hes_low_seed, seed, loss_1 * 1.0 / (i+1), loss_2 * 1.0 / (i+1),  total_loss * 1.0 / (i+1)))
            # 测试
            
            if i % delta_step == 0 and i != 0:
                (exp_batch_obs, exp_batch_action, exp_batch_reward, exp_batch_next_obs, exp_batch_done, exp_batch_last_action, exp_batch_trend) = \
                    exp_gen_rpm_test.exp_fast_learn_sample(EXP_GEN_LEARN_BATCH_SIZE)
                # n_exp_batch_obs = exp_batch_obs
                n_exp_batch_obs = agents[variable_name].normalizerBatch(exp_batch_obs, exp_gen_rpm_test)
                
                sum_hit_cnt = 0
                total_hit_ratio_single = 0.0
                for j in range(EXP_GEN_LEARN_BATCH_SIZE):
                    
                    pred_action = agents[variable_name].get_trend(n_exp_batch_obs[j], np.array([exp_batch_last_action[j]]).astype('float32'))
                    if USE_DOUBLE_NETWORK:
                        pred_action_double = agents_double[variable_name].get_trend(n_exp_batch_obs[j], np.array([exp_batch_last_action[j]]).astype('float32'))
                    last_action = exp_batch_last_action[j]
                    exp_trend = exp_batch_trend[j]
                    exp_trend = exp_trend - 1.0
                    pred_action = pred_action - 1.0
                    
                    
                    
                    
                    if USE_LD_ADAPTOR:
                        pred_action = ld_adaptor.transform(pred_action)
                        if USE_DOUBLE_NETWORK:
                            pred_action_double = env.ld_adaptor_double.transform(pred_action_double)
                    # print("[FL DUBEG] pred_action = ", (list)(pred_action))
                    # print("[FL DUBEG] exp_trend = ", (list)(exp_trend))
                    
                    # delta_action = pred_action - last_action
                    delta_action = pred_action
                    if USE_DOUBLE_NETWORK:
                        # delta_action = (pred_action + pred_action_double) * 0.5
                        delta_action = pred_action * globalValue.DOUBLE_RATIO + pred_action_double * (1-globalValue.DOUBLE_RATIO)
                        
                    sign_delta_action = np.sign(delta_action)
                    product_sign = np.multiply(sign_delta_action, exp_trend)
                    product_sign = np.sign(product_sign)                        
                    # hit_cnt = np.sum(product_sign, axis=0)
                    # if hit_cnt > 0:
                    #     sum_hit_cnt += 1
                    hit_cnt = np.sum(product_sign, axis=0)
                    abs_cnt = np.sum(np.abs(product_sign), axis=0)
                    hit_r_single = hit_cnt * 1.0 / abs_cnt
                    # print("[FL DUBEG] hit_r_single = ", hit_r_single)
                    total_hit_ratio_single += hit_r_single
                    
                    if hit_r_single > EXP_GEN_HIT_THRES:
                        sum_hit_cnt += 1
                    
                test_rate = sum_hit_cnt / EXP_GEN_LEARN_BATCH_SIZE
                print("[FL DUBEG] total_hit_ratio_single = ", total_hit_ratio_single)
                print("[FAST LEARN {0}][PRE_SEED={1}][SEED={2}] evaluate USE_EXP_GEN_FAST_LERAN use gen test, rate = {3}, rate_single = {4}".format(i, self.hes_low_seed, seed, sum_hit_cnt / EXP_GEN_LEARN_BATCH_SIZE, total_hit_ratio_single / EXP_GEN_LEARN_BATCH_SIZE))
                
        return total_hit_ratio_single / EXP_GEN_LEARN_BATCH_SIZE
              
      
    def grid_search_for_init_s(self, seed, act_dim, obs_dim, exp_gen_rpm_train, exp_gen_rpm_test, env):
        for seed_try in range(seed-globalValue.GRID_SERACH_AMPL, seed+globalValue.GRID_SERACH_AMPL+1, 1):
            print("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx [GRID SEARCH][START] xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            print("[GRID SEARCH] seed try = {0}, testing...".format(seed_try))
            print("[GRID SEARCH] act_dim = ", act_dim)
            rate = self.grid_search_for_seed(seed_try, act_dim, obs_dim, exp_gen_rpm_train, exp_gen_rpm_test, env)
            print("[GRID SEARCH] seed try = {0}, rate = {1}".format(seed_try, rate))
            
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  [GRID SEARCH][END]  xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n")
    
    
    def grid_search_for_seed_single_process(self, seed,  result_dict):
        print("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx [GRID SEARCH][START] xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        matuner = MATuner("RL", "SAC_2")
        rate = matuner.grid_search_for_seed_without_data(seed)
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  [GRID SEARCH][END]  xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n")
        # 将结果存储到共享字典中
        result_dict[seed] = rate  # 用 seed 作为键，rate 作为值
        return seed, rate 
    
    
    def grid_search_for_seed_single_process_for_double(self, pre_seed, seed,  result_dict):
        print("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx [GRID SEARCH][START] xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        matuner = MATuner("RL", "SAC_2")
        rate = matuner.grid_search_for_seed_without_data_for_double(pre_seed, seed)
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  [GRID SEARCH][END]  xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n")
        # 将结果存储到共享字典中
        result_dict[seed] = rate  # 用 seed 作为键，rate 作为值
        return seed, rate  
      
    # parallel
    def grid_search_parallel(self, seed):
        # 使用 Manager 创建共享字典
        with multiprocessing.Manager() as manager:
            result_dict = manager.dict()  # 用于存储结果
            
            # 创建进程列表
            processes = []
            for seed_try in range(seed - globalValue.GRID_SERACH_AMPL, seed + globalValue.GRID_SERACH_AMPL + 1, 1):
                process = multiprocessing.Process(
                    target=self.grid_search_for_seed_single_process, 
                    args=(seed_try,  result_dict), 
                    name=f"Process-{seed_try}"
                )
                processes.append(process)
                process.start()

            # 等待所有进程完成
            for process in processes:
                process.join()

            print("All processes have completed.")
            print("Results:", dict(result_dict))  # 将结果转换为普通字典并打印
            return dict(result_dict)  # 返回结果字典
        
    # serialization
    def grid_search_parallel_0(self, seed):
        # 创建结果字典
        result_dict = {}

        # 遍历所有需要尝试的 seed 值
        for seed_try in range(seed - globalValue.GRID_SERACH_AMPL, seed + globalValue.GRID_SERACH_AMPL + 1, 1):
            print(f"Processing seed: {seed_try}")  # 打印当前处理的 seed

            # 调用单个 seed 的搜索函数，直接在当前进程中执行
            try:
                seed_t, rate = self.grid_search_for_seed_single_process(seed_try, result_dict)  # 串行执行
                result_dict[seed_try] = rate  # 存储结果
            except Exception as e:
                print(f"Error processing seed {seed_try}: {str(e)}")  # 异常处理
                result_dict[seed_try] = None  # 设置失败的 seed 结果为空

        # 打印和返回最终结果
        print("All seeds have been processed.")
        print("Results:", result_dict)  # 打印结果字典
        return result_dict  # 返回结果字典
    
    # def grid_search_parallel(self, seed):
    #     # 获取 CPU 核心数，并设置最大并发数
    #     max_processes = min(globalValue.GRID_SERACH_AMPL * 2 + 1, os.cpu_count())

    #     # 构造待处理的 seed 列表
    #     seeds = range(seed - globalValue.GRID_SERACH_AMPL, 
    #                   seed + globalValue.GRID_SERACH_AMPL + 1)

    #     # 使用进程池限制并发数
    #     with multiprocessing.Pool(processes=max_processes) as pool:
    #         # 并行执行任务
    #         results = pool.map(self.grid_search_for_seed_single_process, seeds)

    #     # 处理结果
    #     result_dict = {seed_try: rate for seed_try, rate in results}

    #     print("All processes have completed.")
    #     print("Results:", result_dict)  # 打印结果
    #     return result_dict

    # parallel
    def grid_search_parallel_for_double(self, seed):
        # 使用 Manager 创建共享字典
        with multiprocessing.Manager() as manager:
            result_dict = manager.dict()  # 用于存储结果
            
            # 创建进程列表
            processes = []
            for seed_try in range(seed - globalValue.GRID_SERACH_AMPL, seed + globalValue.GRID_SERACH_AMPL + 1, 1):
                process = multiprocessing.Process(
                    target=self.grid_search_for_seed_single_process_for_double, 
                    args=(seed, seed_try,  result_dict), 
                    name=f"Process-{seed_try}"
                )
                processes.append(process)
                process.start()

            # 等待所有进程完成
            for process in processes:
                process.join()

            print("All processes have completed.")
            print("Results:", dict(result_dict))  # 将结果转换为普通字典并打印
            return dict(result_dict)  # 返回结果字典

    # serialization
    def grid_search_parallel_for_double_0(self, seed):
        # 创建结果字典
        result_dict = {}

        # 遍历所有需要尝试的 seed 值
        for seed_try in range(seed - globalValue.GRID_SERACH_AMPL, seed + globalValue.GRID_SERACH_AMPL + 1, 1):
            print(f"Processing seed: {seed}, seed_try: {seed_try}")  # 打印当前处理的 seed 和 seed_try

            # 调用单个 seed 的搜索函数，直接在当前进程中执行
            try:
                seed_t, rate = self.grid_search_for_seed_single_process_for_double(seed, seed_try, result_dict)  # 串行执行
                result_dict[seed_try] = rate  # 存储结果
            except Exception as e:
                print(f"Error processing seed {seed} and seed_try {seed_try}: {str(e)}")  # 异常处理
                result_dict[seed_try] = None  # 设置失败的 seed_try 结果为空

        # 打印和返回最终结果
        print("All seeds have been processed.")
        print("Results:", result_dict)  # 打印结果字典
        return result_dict  # 返回结果字典



    def evaluate_ce(self, env = None, agent = None, rpm = None, pca = None, TD3logger = None):
        
        if agent == None:   agent = self.agent
        if env == None:     env = self.env
        if rpm == None:     rpm = self.rpm
        if pca == None:     pca = self.pca
        if TD3logger == None:      TD3logger = self.TD3_logger
        
        # eval_reward = []
        total_reward = 0
        env.state = 2
        max_reward = -1
        recommand_action = ''
        globalValue.EVAL_TEST = True
        done = False
        env.start_time = time.time()
        for i in range(1):
            reset_val = True
            while reset_val:
                obs, reset_val = env.reset()
            rear_obs = obs
            if env.info == 'CE':
                TD3logger.info("\n[{} Env initialized][qps: {}, hit_ratio: {}, buffer_size: {}]".format(
                    env.method, env.qps_t0, env.hit_t0, env.bp_size_0))
            elif env.info == 'NODES':
                s = '[' + env.method + 'EVAL Env initialized]'
                s += '[ ' + str(env.se_num) + ' ses:'
                for se in env.se_info:
                    s += '{se' + str(se.uuid) + ', hit_ratio: ' + str(se.hit_t0) + ', buffer_pool_size: ' + str(se.bp_size_t0) + '}'
                s += '][ ' + str(env.ce_num) + ' ces:'
                for ce in env.ce_info:
                    if ce.is_primary == True:
                        s += '{ce' + str(ce.uuid) + ', qps: ' + str(ce.qps_t0) + ', hit_ratio: ' + str(
                            ce.hit_t0) + ', buffer_pool_size: ' + str(ce.bp_size_t0) + '}'
                    else:
                        s += '{ce' + str(ce.uuid) + ', hit_ratio: ' + str(
                            ce.hit_t0) + ', buffer_pool_size: ' + str(ce.bp_size_t0) + '}'
                s += ']'
                TD3logger.info("\n{}".format(s))

            # 给负载预热
            # time.sleep(10)
            steps = 1
            env.eval += 1
            # f = open("./1ce/recommand_knobs/knobs04.txt", 'a')
            # f.write('=======================The recommand knobs==========================')
            # timestamp = get_timestamp()
            # date_str = time_to_str(timestamp)
            # f.write(date_str)
            # f.write("\n")
            # f.close()

            while True:
                print('==========>Eval{}-step{} '.format(env.eval, steps))
                batch_obs = obs
                if done:
                    batch_obs = rear_obs
                done = False
                if TWO_PHASE == True and USE_STATUS_DR == True and not USE_FIXED_DR_KNOBS:
                    batch_obs = pca.transform(batch_obs.reshape(1, -1))
                batch_obs = np.array(batch_obs).flatten()
                # 输入agent时对state标准化
                input_obs = agent.normalizer(batch_obs, rpm)
                # print("Normalize obs:", input_obs)

                if env.method != 'DQN':
                    # if env.method == 'SAC_2':
                    #     action = agent.predict(input_obs.astype('float32'))
                    action_pred = agent.predict(input_obs.astype('float32'), np.array([env.all_last_action]).astype('float32'))
                    action = action_pred
                    if USE_LD_ADAPTOR:
                        action = env.ld_adaptor.transform(action_pred)
                    action = np.clip(action, -1.0, 1.0)
                else:
                    raw_action = agent.sample(input_obs.astype('float32'))
                    action = env.explian_DQN_action(raw_action)

                # bpsize一次的变化最好不要超过2或1/2，但env reset时无需遵循此规则
                if env.info == 'CE':
                    if env.last_action != -2:
                        bp_size = action_mapping(action[0], env.min_info[0], env.max_info[0])
                        # np.random.seed(time.time())
                        if bp_size > env.last_bp_size * 2:
                            # 如果当前预测值大于2倍上次值，则在[last_size,2*last_size]中随机一个值
                            # print("random.randint big")
                            bp_size = np.random.randint(env.last_bp_size, env.last_bp_size * 2)
                        elif bp_size < env.last_bp_size * 0.5:
                            # 如果当前预测值小于0.5倍上次值，则在[0.5*last_size,last_size]中随机一个值
                            # print("random.randint small")
                            bp_size = np.random.randint(env.last_bp_size * 0.5, env.last_bp_size)
                        action[0] = utils.real_action_to_action(bp_size, env.min_info[0], env.max_info[0])
                elif env.info == 'NODES':
                    if env.last_action != 0:
                        # 依次修改node
                        index = 0
                        for se in env.se_info:
                            # buf_key = se.uuid + '#buffer_pool_size'
                            bp_size_se = action_mapping(action[index], se.tune_action['buffer_pool_size'][1], se.tune_action['buffer_pool_size'][2])
                            last_se_bp_size = se.last_bp_size
                            if bp_size_se > last_se_bp_size * 2:
                                # 如果当前预测值大于2倍上次值，则在[last_size,2*last_size]中随机一个值
                                # print("random.randint big")
                                bp_size_se = np.random.randint(last_se_bp_size, last_se_bp_size * 2)
                            elif bp_size_se < last_se_bp_size * 0.5:
                                # 如果当前预测值小于0.5倍上次值，则在[0.5*last_size,last_size]中随机一个值
                                # print("random.randint small")
                                bp_size_se = np.random.randint(last_se_bp_size * 0.5, last_se_bp_size)
                            action[index] = utils.real_action_to_action(bp_size_se,
                                                                        se.tune_action['buffer_pool_size'][1],
                                                                        se.tune_action['buffer_pool_size'][2])
                            index += len(se.tune_action)

                        for ce in env.ce_info:
                            # buf_key = ce.uuid + '#buffer_pool_size'
                            bp_size_ce = action_mapping(action[index], ce.tune_action['buffer_pool_size'][1],
                                                        ce.tune_action['buffer_pool_size'][2])
                            last_ce_bp_size = ce.last_bp_size
                            # ce
                            if bp_size_ce > last_ce_bp_size * 2:
                                # 如果当前预测值大于2倍上次值，则在[last_size,2*last_size]中随机一个值
                                # print("random.randint big")
                                bp_size_ce = np.random.randint(last_ce_bp_size, last_ce_bp_size * 2)
                            elif bp_size_ce < last_ce_bp_size * 0.5:
                                # 如果当前预测值小于0.5倍上次值，则在[0.5*last_size,last_size]中随机一个值
                                # print("random.randint small")
                                bp_size_ce = np.random.randint(last_ce_bp_size * 0.5, last_ce_bp_size)
                            action[index] = utils.real_action_to_action(bp_size_ce,
                                                                        ce.tune_action['buffer_pool_size'][1],
                                                                        ce.tune_action['buffer_pool_size'][2])
                            index += len(ce.tune_action)

                # print('action after filter: ', action)

                TD3logger.info("\n[{}] Action: {}".format(env.method, action))

                # #action = [action]
                # action_record = utils.action_change(env, action)
                # # 将推荐参数封装为发送数据标准格式
                # if env.info == 'CE':
                #     var_names = list(env.all_actions.keys())
                #     send_variables = utils.get_set_variables_string(var_names, action_record, False, 3)
                # elif env.info == 'NODES':
                #     var_names = list(env.all_actions.keys())
                #     send_variables_se, send_variables_ce = utils.get_set_variables_string_nodes(var_names, action_record, 3)
                #     send_variables = send_variables_se + ' ' + send_variables_ce

                steps += 1
                next_obs, reward, done, info = env.step(action)
                
                if env.ld_adaptor is not None:
                    action_pred = env.ld_adaptor.reverse_transform(action)
                    env.all_last_action_2 = env.all_last_action
                    env.all_last_action = action_pred
                
                
                if done:
                    continue
                rear_obs = obs
                obs = next_obs
                total_reward += reward

                if max_reward < reward:
                    max_reward = reward
                    # recommand_action = send_variables

                if env.info == 'CE':
                    TD3logger.info(
                        "\n[{}][Eval: {}][Step: {}][Metric qps:{} hit_ratio:{} buffer_size:{}]Reward: {} Score: {} Done: {}".format(
                            env.method, env.eval, steps, env.last_qps, env.last_hr, env.last_bp_size, reward,
                            total_reward, done
                        ))
                elif env.info == 'NODES':
                    s = '[' + env.method + '][Eval: ' + str(env.eval) + '][Step: ' + str(steps) + ']'
                    s += '[ ' + str(env.se_num) + ' ses:'
                    for se in env.se_info:
                        s += '{se' + str(se.uuid) + ', hit_ratio: ' + str(se.last_hr) + ', buffer_pool_size: ' + str(se.last_bp_size) + '}'
                    s += '][ ' + str(env.ce_num) + ' ces:'
                    for ce in env.ce_info:
                        if ce.is_primary == True:
                            s += '{ce' + str(ce.uuid) + ', qps: ' + str(ce.last_qps) + ', hit_ratio: ' + str(
                                ce.last_hr) + ', buffer_pool_size: ' + str(ce.last_bp_size) + '}'
                        else:
                            s += '{ce' + str(ce.uuid) + ', hit_ratio: ' + str(
                                ce.last_hr) + ', buffer_pool_size: ' + str(ce.last_bp_size) + '}'
                    s += ']'
                    s += 'Reward: ' + str(reward) + ', Score: ' + str(total_reward) + ', Done: ' + str(done)
                    TD3logger.info("\n{}".format(s))

                if done or steps >= MOVE_STEPS:
                    # env.unuse_step = 0
                    break
            # eval_reward.append(total_reward)
        # return np.mean(eval_reward)
        #
        # f = open("./1ce/recommand_knobs/knobs04.txt", 'a')
        # f.write(recommand_action)
        # f.write('        reward = ')
        # f.write(str(max_reward))
        # f.write("\n")
        # f.close()
        globalValue.EVAL_TEST = False
        env.state = 1
        env.end_time = time.time()
        return total_reward

    def evaluate_t(self, flag):
        if flag:
            print('------SE Train thread start...------')
        else:
            print('------CE Train thread start...------')
        if flag:
            env = SEEnv()
        else:
            env = CEEnv()

        #
        # obs_dim = env.state_dim
        # act_dim = env.action_dim
        # globalValue.EVAL_TEST = True
        #
        # model = Model(act_dim)
        # algorithm = DDPG(
        #     model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
        # agent = Agent(algorithm, obs_dim, act_dim)

        obs_dim = env.state_dim
        act_dim = env.action_dim
        max_action = 1.0

        model = TestModel(act_dim, max_action)
        algorithm = TD3(
            model,
            max_action=max_action,
            gamma=GAMMA,
            tau=TAU,
            actor_lr=ACTOR_LR,
            critic_lr=CRITIC_LR,
            policy_noise=POLICY_NOISE,  # Noise added to target policy during critic update
            noise_clip=NOISE_CLIP,  # Range to clip target policy noise
            policy_freq=POLICY_FREQ
        )
        agent = TestAgent(algorithm, obs_dim, act_dim)

        rpm = None
        if not USE_PRIORITY_RPM:
            ld = act_dim
            hd = act_dim
            if USE_LD_ADAPTOR:
                ld = env.ld_adaptor.target_dim
                hd = env.ld_adaptor.high_dim
            rpm = ReplayMemory(MEMORY_SIZE, act_dim, obs_dim, USE_LD_ADAPTOR, ld, hd)
        else:
            rpm = PrioritizedReplayMemory(MEMORY_SIZE, act_dim, obs_dim)


        # 导入已有模型
        agent.restore("/home/fox/subject/delta_qps/tune (复件)/1ce/model_dir/ce_steps_1617630080.ckpt")


        if flag:
            f = open("./1se/rpm_dir/se_rpm.txt", "rb")
        else:
            f = open("./1ce/rpm_dir/ce_rpm_full0330.txt", "rb")
        rpm = pickle.load(f)
        f.close()

        eval_reward = self.evaluate_ce(env, agent)
        print('Test reward:{}'.format(eval_reward))









   
   






