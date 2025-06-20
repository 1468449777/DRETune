# 可学习的专家奖励网络 LERC Learned Expert Reward Controller
# 输入：当前动作、初始动作
# 输出：对于每个动作的打分向量 / 总的奖励

# 模块架构：
#       1. 趋势计算模块：根据当前动作和初始动作计算出 变化趋势向量 trend_vec 和 动作变化量向量 delta_val_vec
#       2. 网络模块：输入 初始动作向量 init_action_vec、动作变化量向量 delta_val_vec 、 环境 obs ，输出 打分向量 exp_reward_vec
#       3. 奖励计算模块：最终 exp_reward = sum( trend_vec * exp_reward_vec )

# 更新方式：根据环境真实的奖励构造误差，更新网络参数


import parl
from parl import layers
import numpy as np
from maEnv import datautils
from paddle import fluid
import os
from parl.core.fluid.algorithm import Algorithm
import random
import globalValue


SINGLE_NODE_OBS_DIM = 16

ENABLE_DEBUG_INFO = False
LERC_LR = 0.001


# LERC model
class LERCModel(parl.Model):
    def __init__(self, action_dim):
        # 定义模型结构
        hid1_size = 256
        hid2_size = 512
        hid3_size = 1024
        hid4_size = 512
        hid5_size = 256

        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=hid3_size, act='relu')
        self.fc4 = layers.fc(size=hid4_size, act='relu')
        self.fc5 = layers.fc(size=hid5_size, act='relu')
        self.output = layers.fc(size=action_dim, act='sigmoid')

    def forward(self, act):
        # 前向传播定义
        hid1 = self.fc1(act)
        hid2 = self.fc2(hid1)
        hid3 = self.fc3(hid2)
        hid4 = self.fc4(hid3)
        hid5 = self.fc5(hid4)
        output = self.output(hid5)

        return output


# LERC
class LERC():
    def __init__(
        self, 
        obs_dim, 
        action_dim, 
        use_learn=False,
        batch_size=16, 
        node_num = 2, 
        use_ld=False, 
        ld_adaptor=None
    ):
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.model = LERCModel(self.action_dim)
        self.alg = LERCAlg(
            lerc_model=self.model,
            lerc_lr=LERC_LR,
            use_learn=use_learn,
            batch_size=batch_size,
            act_dim=action_dim,
            obs_dim=obs_dim
        )
        self.agent = LERCAgent(
            algorithm=self.alg,
            act_dim=self.action_dim,
            obs_dim=self.obs_dim
        )
        
        self.node_num = node_num
        self.use_ld = use_ld
        self.ld_adaptor = ld_adaptor

     # TODO: 需要结合状态，状态怎么获取？
    # input: s_ratio 状态比例：labelA、B、C...比例
    # input: r_ratio 奖励/动作比例：正向样本:负向样本:0样本
    # 0     location += tune_get_single_data(buf+location,&total_info->free_list_len,tune_ulint);
    # 1     location += tune_get_single_data(buf+location,&total_info->lru_len,tune_ulint);
    # 2     location += tune_get_single_data(buf+location,&total_info->old_lru_len,tune_ulint);
    # 3     location += tune_get_single_data(buf+location,&total_info->flush_list_len,tune_ulint);
    # 4     location += tune_get_single_data(buf+location,&total_info->n_pend_reads,tune_ulint);
    # 5     location += tune_get_single_data(buf+location,&total_info->n_pending_flush_lru,tune_ulint);
    # 6     location += tune_get_single_data(buf+location,&total_info->n_pending_flush_list,tune_ulint);
    # 7     location += tune_get_single_data(buf+location,&total_info->io_cur,tune_int);
    # 8     location += tune_get_single_data(buf+location,&activity_count,tune_int);
    # 9     location += tune_get_single_data(buf+location,&total_info->n_page_get_delta,tune_ulint);
    # 10    location += tune_get_single_data(buf+location,&total_info->pages_read_rate,tune_double);
    # 11    location += tune_get_single_data(buf+location,&total_info->pages_created_rate,tune_double);
    # 12    location += tune_get_single_data(buf+location,&total_info->pages_written_rate,tune_double);
    # 13    location += tune_get_single_data(buf+location,&total_info->pages_evicted_rate,tune_double);
    def generate_expert_exp_old(self, s_ratio, r_ratio, total_num, env):
        
        s_list=[]
        a_list=[]
        l_a_list=[]
        r_list = []
        d_list = np.zeros(total_num, dtype=float)
        n_s_list = []
        
        
        # 状态生成
        knobs_range = [
            ["int", 0, 10000],          # 0     free_list_len
            ["int", 0, 10000],          # 1     lru_len
            ["int", 0, 10000],          # 2     old_lru_len
            ["int", 0, 10000],          # 3     flush_list_len
            ["int", 0, 10000],          # 4     n_pend_reads
            ["int", 0, 10000],          # 5     n_pending_flush_lru
            ["int", 0, 10000],          # 6     n_pending_flush_list
            ["int", -10000, 10000],     # 7     io_cur
            ["int", -10000, 10000],     # 8     activity_count
            ["int", 0, 10000],          # 9     n_page_get_delta
            ["double", 0.0, 1.0],       # 10    pages_read_rate
            ["double", 0.0, 1.0],       # 11    pages_created_rate
            ["double", 0.0, 1.0],       # 12    pages_written_rate
            ["double", 0.0, 1.0]        # 13    pages_evicted_rate
            # you can add more ranges here
        ]
        # 专家规则中涉及到的状态，按照比例提前生成好
        expert_s_idx = [10, 12, 3, 1, 6, 0, 11, 7]
        
        # 生成符合 label 比例的 status
        
        s_list = np.zeros((total_num * self.node_num, int(self.obs_dim / self.node_num)), dtype=float)
        
        s_r_A = s_ratio[0]
        s_r_B = s_r_A + s_ratio[1]
        s_r_C = s_r_B + s_ratio[2]
        
        # label A
        for i in range(total_num * self.node_num):
            rnd_t = random.random()
            
            if rnd_t < s_r_A / 3.0:
                # 满足 1. data_list[10] < data_list[12]
                s_list[i][10] = random.uniform(0, 1)
                s_list[i][12] = random.uniform(s_list[i][10], 1)
                # 不满足 2. data_list[3] > 0.5 * data_list[1]
                s_list[i][1] = random.randint(0, 10000)
                s_list[i][3] = random.randint(0, int(0.5 * s_list[i][1]))
                # 不满足 3. data_list[6] > 0
                s_list[i][6] = 0
                
            elif rnd_t >= s_r_A / 3.0 and rnd_t < 2 * s_r_A / 3.0:
                # 不满足 1. data_list[10] < data_list[12]
                s_list[i][10] = random.uniform(0, 1)
                s_list[i][12] = random.uniform(0, s_list[i][10])
                # 满足 2. data_list[3] > 0.5 * data_list[1]
                s_list[i][1] = random.randint(0, 10000)
                s_list[i][3] = random.randint(int(0.5 * s_list[i][1]), 10000)
                # 不满足 3. data_list[6] > 0
                s_list[i][6] = 0
                
            elif rnd_t >= 2 * s_r_A / 3.0 and rnd_t < s_r_A:
                # 不满足 1. data_list[10] < data_list[12]
                s_list[i][10] = random.uniform(0, 1)
                s_list[i][12] = random.uniform(0, s_list[i][10])
                # 不满足 2. data_list[3] > 0.5 * data_list[1]
                s_list[i][1] = random.randint(0, 10000)
                s_list[i][3] = random.randint(0, int(0.5 * s_list[i][1]))
                # 满足 3. data_list[6] > 0
                s_list[i][6] = random.randint(1, 10000)
            
            else:
                # 不满足 1. data_list[10] < data_list[12]
                s_list[i][10] = random.uniform(0, 1)
                s_list[i][12] = random.uniform(0, s_list[i][10])
                # 不满足 2. data_list[3] > 0.5 * data_list[1]
                s_list[i][1] = random.randint(0, 10000)
                s_list[i][3] = random.randint(0, int(0.5 * s_list[i][1]))
                # 不满足 3. data_list[6] > 0
                s_list[i][6] = 0
            
        # label B
        for i in range(total_num * self.node_num):
            rnd_t = random.random()
            
            if rnd_t >= s_r_A and rnd_t < s_r_B:
                # 满足 1. data_list[0] == 0
                s_list[i][0] = 0
                # 满足 2. data_list[11] > 0
                s_list[i][11] = random.uniform(0.001, 1)
            else:
                if rnd_t < (s_r_A + s_r_B) / 3.0:
                    # 不满足 1. 满足 2.
                    s_list[i][0] = random.randint(1, 10000)
                    s_list[i][11] = random.uniform(0.001, 1)
                elif rnd_t >= (s_r_A + s_r_B) / 3.0 and rnd_t < 2 * (s_r_A + s_r_B) / 3.0:
                    # 不满足 2. 满足 1.
                    s_list[i][0] = 0
                    s_list[i][11] = 0.0
                else:
                    # 不满足 1. 2.
                    s_list[i][0] = random.randint(1, 10000)
                    s_list[i][11] = 0.0
        
        # label C
        for i in range(total_num * self.node_num):
            rnd_t = random.random()
            if rnd_t >= s_r_B and rnd_t <= s_r_C:
                s_list[i][7] = random.randint(200, 10000)
            else:
                s_list[i][7] = random.randint(-10000, 200)
                
        # 生成其他状态参数
        for j in range(int(self.obs_dim / self.node_num)):
            if j not in expert_s_idx:
                for i in range(total_num * self.node_num):
                    if knobs_range[j][0] == "int":
                        s_list[i][j] = random.randint(knobs_range[j][1], knobs_range[j][2])
                    elif knobs_range[j][0] == "double":
                        s_list[i][j] = random.uniform(knobs_range[j][1], knobs_range[j][2])
        
        # 组合多个节点的状态参数
        # print("s_list = ", s_list.shape)
        node_s = np.array_split(s_list, self.node_num)
        new_s_list = np.concatenate(node_s, axis=1)
        # print("new_s_list = ", new_s_list.shape)
        # print(new_s_list)
        
        
        
        # 上一次动作生成
        l_a_list = np.random.uniform(low=-0.99, high=0.99, size=(total_num, self.action_dim))


        # 根据专家知识生成相应动作和奖励
        # 获取 trend 
        self.get_trend(new_s_list, env)
        # print(self.batch_labels)
        # print(self.batch_trends)
        # 奖励放缩系数
        exp_gen_r_scale = 0.2
        
        # 根据trend和r_ratio生成action
        a_list = np.zeros((total_num, self.action_dim), dtype=float)
        r_list = np.zeros(total_num, dtype=float)
        pos_p = r_ratio[0]
        neg_p = pos_p + r_ratio[1]
        for i in range(total_num):
            r_cnt = 0
            for j in range(self.action_dim):
                rnd_t = random.random()
                if rnd_t < pos_p:
                    r_cnt += 1
                    if self.batch_trends[i][j] == 1:
                        a_list[i][j] = random.uniform(l_a_list[i][j], 1.0)
                    elif self.batch_trends[i][j] == -1:
                        a_list[i][j] = random.uniform(-1.0, l_a_list[i][j])
                    else:
                        a_list[i][j] = random.uniform(-1.0, 1.0)
                elif rnd_t < neg_p:
                    r_cnt -= 1
                    if self.batch_trends[i][j] == 1:
                        a_list[i][j] = random.uniform(-1.0, l_a_list[i][j])
                    elif self.batch_trends[i][j] == -1:
                        a_list[i][j] = random.uniform(l_a_list[i][j], 1.0)
                    else:
                        a_list[i][j] = random.uniform(-1.0, 1.0)
            r_list[i] = exp_gen_r_scale * r_cnt
            
        # print("r_list = ", r_list)
        # print("a_list = ", a_list)
        
        if self.use_ld:
            for i in range(total_num):
                self.batch_trends[i] = self.ld_adaptor.reverse_transform(self.batch_trends[i])
        
                    
        # 下一个状态无用，随便生成
        n_s_list = np.zeros((total_num, self.obs_dim), dtype=float)
        
        
        
        
        # batch_obs, batch_action, batch_reward, batch_next_obs, batch_done
        return new_s_list, a_list, np.array(r_list), n_s_list, d_list, l_a_list, np.array(self.batch_trends)
            
    
    
    
    # 生成规则模拟经验的主要函数
    def generate_expert_exp(self, total_num, env):
        
        my_rand = np.random.RandomState(globalValue.FAST_LEARN_GEN_SEED)
        
        s_list=[]
        a_list=[]
        l_a_list=[]
        r_list = []
        d_list = np.zeros(total_num, dtype=float)
        n_s_list = []
        
        
        # 状态生成
        knobs_range = [
            ["int", 0, 10000],          # 0     free_list_len
            ["int", 0, 10000],          # 1     lru_len
            ["int", 0, 10000],          # 2     old_lru_len
            ["int", 0, 10000],          # 3     flush_list_len
            ["int", 0, 10000],          # 4     n_pend_reads
            ["int", 0, 10000],          # 5     n_pending_flush_lru
            ["int", 0, 10000],          # 6     n_pending_flush_list
            ["int", -10000, 10000],     # 7     io_cur
            ["int", -10000, 10000],     # 8     activity_count
            ["int", 0, 10000],          # 9     n_page_get_delta
            ["double", 0.0, 1.0],       # 10    pages_read_rate
            ["double", 0.0, 1.0],       # 11    pages_created_rate
            ["double", 0.0, 1.0],       # 12    pages_written_rate
            ["double", 0.0, 1.0]        # 13    pages_evicted_rate
            # you can add more ranges here
        ]
        
        
        
        
        s_total_num = (int)(total_num / globalValue.FAST_LEARN_A_PER_S)
        s_list = np.zeros((s_total_num * self.node_num, int(self.obs_dim / self.node_num)), dtype=float)
        
        # 生成状态参数
        primary_idx = len(globalValue.CONNECT_SE_IP)
        # print(s_total_num, total_num)
        for j in range(int(self.obs_dim / self.node_num)):
            for i in range(s_total_num * self.node_num):
                if j < 14 and knobs_range[j][0] == "int":
                    s_list[i][j] = my_rand.randint(knobs_range[j][1], knobs_range[j][2])
                elif j < 14 and knobs_range[j][0] == "double":
                    s_list[i][j] = my_rand.uniform(knobs_range[j][1], knobs_range[j][2])
            
                # delta_q0
                elif j == 14:
                    # if i == primary_idx:
                    #     s_list[i][j] = my_rand.uniform(-3.0, 3.0)
                    # else:
                    if globalValue.USE_PROXY_SQL:
                        s_list[i][j] = my_rand.uniform(-1.0, 1.0)
                    else:
                        s_list[i][j] = 0.0
                # delta_h0
                elif j == 15:
                    s_list[i][j] = my_rand.uniform(-0.5, 0.5)
            
            
        # 组合多个节点的状态参数
        # print("[S DEBUG] s_list = ", s_list)
        node_s = np.array_split(s_list, self.node_num)
        
        # print("[S DEBUG] node_s = ", node_s)
        
        
        new_s_t = np.concatenate(node_s, axis=1)
        # print("[S DEBUG] new_s_t = ", new_s_t)
        
        # primary delta_q0
        if not globalValue.USE_PROXY_SQL:
            for i in range(new_s_t.shape[0]):
                qps_idx = SINGLE_NODE_OBS_DIM * (len(globalValue.CONNECT_SE_IP)) + 14
                new_s_t[i][qps_idx] = my_rand.uniform(-1.0, 1.0)
            
        # print("[S DEBUG] new_s_t = ", new_s_t)
        
        
        
        # print("new_s_t = ", new_s_t.shape)
        # print("[FL GEN DEBUG] new_s_list shape = ", new_s_list.shape)
        # new_s_list = np.tile(new_s_list_t, (100, 1))  # 沿第一个维度重复 100 次
        
        # 用 list comprehension 创建包含 100 个 s 的列表
        new_s_list_t = [new_s_t for _ in range(globalValue.FAST_LEARN_A_PER_S)]
        

        # 使用 np.concatenate 将这些数组沿第一个轴拼接
        new_s_list = np.concatenate(new_s_list_t, axis=0)
        
        
        
        
        # 上一次动作生成
        l_a_list_ld = my_rand.uniform(low=-0.99, high=0.99, size=(total_num, self.action_dim))
        a_list = my_rand.uniform(low=-0.3, high=0.3, size=(total_num, self.action_dim))
        
        # 根据专家知识生成相应动作和奖励
        # 获取 trend
        # print("new_s_list = ", new_s_list.shape)
        
        self.get_trend(new_s_list, env)
        # logits test
        self.batch_trends = np.array(self.batch_trends)+1
        # print(self.batch_trends)

        # print("new_s_list = ", new_s_list.shape)
        actual_trend_list = []
        for i in range(total_num):
            l_a_ld = l_a_list_ld[i]
            a_ld = a_list[i]
            if env.ld_adaptor:
                l_a_hd = env.ld_adaptor.transform(l_a_ld)
                a_hd = env.ld_adaptor.transform(a_ld)
            else:
                l_a_hd = l_a_ld
                a_hd = a_ld
                
            diff = a_hd
            # a_list.append(diff)
            l_a_list.append(l_a_hd)
            actual_trend = np.sign(diff)
            # print("i = ", i , ", batch_trends shape = ", len(self.batch_trends))
            product = np.multiply(actual_trend, self.batch_trends[i])
            r = np.sum(product, axis=0)
            # print("\n[FL EXP GEN] s = ", new_s_list[i])
            # print("[FL EXP GEN] labels = ", self.batch_labels[i])
            # print("[FL EXP GEN] trend = ", self.batch_trends[i])
            # print("[FL EXP GEN] actual_trend = ", actual_trend)
            # print("[FL EXP GEN] product = ", product)
            # print("[FL EXP GEN] r = ", r)
            
            
            
            
            r_list.append(r)
            
            
        # 下一个状态无用，随便生成
        n_s_list = np.zeros((total_num, self.obs_dim), dtype=float)
        
        r_list = np.array(r_list)
        # a_list = np.array(a_list)
        l_a_list = np.array(l_a_list)
        trends_list = np.array(self.batch_trends)
        
        print("\n======================[FAST LEARN][GEN]======================")
        print("new_s_list shape = ", new_s_list.shape)
        print("a_list shape = ", a_list.shape)
        print("r_list shape = ", r_list.shape)
        print("n_s_list shape = ", n_s_list.shape)
        print("d_list shape = ", d_list.shape)
        print("l_a_list shape = ", l_a_list.shape)
        print("trends_list shape = ", trends_list.shape)
        print("======================[FAST LEARN][END]======================\n")
        
        
        # batch_obs, batch_action, batch_reward, batch_next_obs, batch_done
        return new_s_list, a_list, r_list, n_s_list, d_list, l_a_list, trends_list  

        
        
        
        








    # def get_trend(self, env):
    #     # TODO: 以后可以做成从外解读取并解析规则，支持以一定格式自定义规则

    #     # 解析每个节点的 label
    #     self.get_nodes_labels(env)

    #     # 根据 label 判断每个节点相应参数的专家推荐的 trend
    #     expert_trends = self.get_expert_trends(env)

    #     # # 计算实际的变化量
    #     # delta_actions = current_action - init_action

    #     if ENABLE_DEBUG_INFO:
    #         print("\n========================== LERC LOG ==========================")
    #         print("[LERC LOG][class:LERC][func:get_trend] expert_trends = {0}".format(expert_trends))
    #         # print("[LERC LOG][class:LERC][func:get_trend] delta_actions = {0}".format(delta_actions))
    #         print("========================== LERC LOG ==========================\n")

    #     return expert_trends
    #         # , delta_actions

    # 获取trend
    def get_trend(self, batch_obs, env):
        self.batch_labels = self.get_labels(batch_obs, env)
        # print("[MERGE DEBUG] nodes lerc labels = {0}".format(self.batch_labels))

        if ENABLE_DEBUG_INFO:
            print("\n========================== LERC LOG ==========================")
            print("[LERC LOG][class:LERC][func:get_trend] batch_labels = {0}".format(
                self.batch_labels))
            print("========================== LERC LOG ==========================\n")

        self.batch_trends = self.all_nodes_labels_to_action_trend(env)
        if ENABLE_DEBUG_INFO:
            print("\n========================== LERC LOG ==========================")
            print("[LERC LOG][class:LERC][func:get_trend] batch_trends = {0}".format(
                self.batch_trends))
            print("========================== LERC LOG ==========================\n")
        return self.batch_trends

    def get_network_reward(self, delta_actions, obs):
        pred_reward_vec = self.agent.predict(delta_actions, obs)
        if ENABLE_DEBUG_INFO:
            print("\n========================== LERC LOG ==========================")
            print("[LERC LOG][class:LERC][func:get_network_reward] pred_reward_vec = {0}".format(
                pred_reward_vec))
            print("========================== LERC LOG ==========================\n")

        return pred_reward_vec

    def get_expert_reward(self, env, batch_obs, batch_delta_actions):
        self.get_trend(batch_obs, env)
        self.batch_pred_reward = self.get_network_reward(
            batch_delta_actions, batch_obs)
        if ENABLE_DEBUG_INFO:
            print("\n========================== LERC LOG ==========================")
            print("[LERC LOG][class:LERC][func:get_expert_reward] batch_pred_reward = {0}".format(
                self.batch_pred_reward))
            print("========================== LERC LOG ==========================\n")

        # 使用 NumPy 的符号函数 np.sign() 来获取每个元素的正负符号（+1, -1, 0）
        signs_actions = np.sign(batch_delta_actions)
        signs_trends = np.sign(self.batch_trends)

        # 创建结果数组，初始化为 -1 （假设不匹配）
        self.r_ratio = np.full_like(batch_delta_actions, -1)

        # 如果符号匹配，设置为1
        self.r_ratio[signs_actions == signs_trends] = 1

        # 如果其中一个数组的符号是0，结果也设置为0
        self.r_ratio[(signs_actions == 0) | (signs_trends == 0)] = 0

        if ENABLE_DEBUG_INFO:
            print("\n========================== LERC LOG ==========================")
            print("[LERC LOG][class:LERC][func:get_expert_reward] batch_delta_actions = {0}".format(
                batch_delta_actions))
            print("[LERC LOG][class:LERC][func:get_expert_reward] batch_trends = {0}".format(
                self.batch_trends))
            print("[LERC LOG][class:LERC][func:get_expert_reward] r_ratio = {0}".format(
                self.r_ratio))
            print("========================== LERC LOG ==========================\n")

        self.expert_reward = self.r_ratio * self.batch_pred_reward
        if np.sum(np.abs(self.r_ratio), axis=1).any() != 0:
            mean_expert_reward = np.sum(
                self.expert_reward, axis=1) / np.sum(np.abs(self.r_ratio), axis=1)
        else:
            mean_expert_reward = np.mean(self.expert_reward, axis=1)

        if ENABLE_DEBUG_INFO:
            print("\n========================== LERC LOG ==========================")
            print("[LERC LOG][class:LERC][func:get_expert_reward] expert_reward = {0}".format(
                self.expert_reward))
            print("[LERC LOG][class:LERC][func:get_expert_reward] mean_expert_reward = {0}".format(
                mean_expert_reward))

            print("========================== LERC LOG ==========================\n")

        return mean_expert_reward

    # def get_nodes_labels(self, env):
    #     for se in env.se_info:
    #         datautils.status_to_labels(se)
    #     for ce in env.ce_info:
    #         datautils.status_to_labels(ce)

    # def get_expert_trends(self, env):
    #     expert_trends = datautils.all_nodes_labels_to_action_trend(env)
    #     return expert_trends

    # 获取标签
    def get_labels(self, batch_obs, env):
        batch_label = []
        for obs in batch_obs:
            label = []
            se_num = len(env.se_info)
            for i in range(len(env.se_info)):
                node_obs = obs[i * SINGLE_NODE_OBS_DIM: (i+1) * SINGLE_NODE_OBS_DIM]
                node_label = self.status_to_labels(node_obs, env.se_info[i])
                # print("lerc label = ", node_label)
                label.append(node_label)

            for i in range(len(env.ce_info)):
                node_obs = obs[(se_num+i) * SINGLE_NODE_OBS_DIM: (se_num+i+1) * SINGLE_NODE_OBS_DIM]
                node_label = self.status_to_labels(node_obs, env.ce_info[i])
                label.append(node_label)
            batch_label.append(label)
        return batch_label

    def node_labels_to_action_trend_old(self, label, info, action_trend, index, node):
        # 解析labelA:0-写密集，1-读密集 ->（1变大，-1变小）
        if label[0] == 0:
            if info == 'old_blocks_time':
                action_trend[index] = -1
            elif info == 'random_read_ahead':
                action_trend[index] = -1
            elif info == 'read_ahead_threshold':
                action_trend[index] = 1
            elif info == 'lock_wait_timeout':
                action_trend[index] = 1
            elif info == 'lru_scan_depth':
                action_trend[index] = 1
            elif info == 'lru_sleep_time_flush':
                action_trend[index] = -1
            elif info == 'flush_n':
                action_trend[index] = 1
        else:
            if info == 'old_blocks_time':
                action_trend[index] = 1
            elif info == 'random_read_ahead':
                action_trend[index] = 1
            elif info == 'read_ahead_threshold':
                action_trend[index] = -1
            elif info == 'lock_wait_timeout':
                action_trend[index] = -1

        # 解析labelB:0-低负载，1-高负载 ->（1变大，-1变小）
        if label[1] == 0:
            if info == 'buffer_pool_size' and not node.is_primary:
                action_trend[index] = -1
                # print(node.name + ' bpsize_trend ' + str(action_trend[index]))
            elif info == 'flushing_avg_loops':
                action_trend[index] = 1
            elif info == 'old_blocks_pct':
                action_trend[index] = -1
        else:
            if info == 'buffer_pool_size':
                action_trend[index] = 1
                # print(node.name + ' bpsize_trend ' + str(action_trend[index]))
            elif info == 'flushing_avg_loops':
                action_trend[index] = -1
            elif info == 'old_blocks_pct':
                action_trend[index] = 1
            elif info == 'ce_coordinator_sleep_time':
                action_trend[index] = -1

        if label[2] == 1 or label[1] == 1:
            if info == 'io_capacity':
                action_trend[index] = 1
        else:
            if info == 'io_capacity':
                action_trend[index] = -1
        # TODO: 不知道为啥符号相反了，很奇怪
        # action_trend[index] = -action_trend[index]

    # 根据标签获取trend（新版，可使用部分规则）
    def node_labels_to_action_trend(self, label, info, action_trend, index, node):
        rule = range(30)
        if globalValue.USE_PARTIAL_RULE:
            rule = globalValue.PARTIAL_RULE
        # 解析labelA:0-写密集，1-读密集 ->（1变大，-1变小）
        if label[0] == 0:
            if info == 'old_blocks_time' and 0 in rule:
                action_trend[index] = -1
            elif info == 'random_read_ahead' and 1 in rule:
                action_trend[index] = -1
            elif info == 'read_ahead_threshold' and 2 in rule:
                action_trend[index] = 1
            elif info == 'lock_wait_timeout' and 3 in rule:
                action_trend[index] = 1
            elif info == 'lru_scan_depth' and 4 in rule:
                action_trend[index] = 1
            elif info == 'lru_sleep_time_flush' and 5 in rule:
                action_trend[index] = -1
            elif info == 'flush_n' and 6 in rule:
                action_trend[index] = 1
        else:
            if info == 'old_blocks_time' and 7 in rule:
                action_trend[index] = 1
            elif info == 'random_read_ahead' and 8 in rule:
                action_trend[index] = 1
            elif info == 'read_ahead_threshold' and 9 in rule:
                action_trend[index] = -1
            elif info == 'lock_wait_timeout' and 10 in rule:
                action_trend[index] = -1

        # 解析labelB:0-低负载，1-高负载 ->（1变大，-1变小）
        if label[1] == 0:
            if info == 'buffer_pool_size' and 11 in rule:
                action_trend[index] = -1
                # print(node.name + ' bpsize_trend ' + str(action_trend[index]))
            elif info == 'flushing_avg_loops' and 12 in rule:
                action_trend[index] = 1
            elif info == 'old_blocks_pct' and 13 in rule:
                action_trend[index] = -1
        else:
            if info == 'buffer_pool_size' and 14 in rule:
                action_trend[index] = 1
                # print(node.name + ' bpsize_trend ' + str(action_trend[index]))
            elif info == 'flushing_avg_loops' and 15 in rule:
                action_trend[index] = -1
            elif info == 'old_blocks_pct' and 16 in rule:
                action_trend[index] = 1
            elif info == 'ce_coordinator_sleep_time' and 17 in rule:
                action_trend[index] = -1
            elif info == 'io_capacity' and 18 in rule:
                action_trend[index] = 1

        if label[2] == 1:
            if info == 'io_capacity' and 19 in rule:
                action_trend[index] = 1
        else:
            if info == 'io_capacity' and 20 in rule:
                action_trend[index] = -1
        # TODO: 不知道为啥符号相反了，很奇怪
        # action_trend[index] = -action_trend[index]




    def all_nodes_labels_to_action_trend(self, env):

        batch_trend = []
        for label in self.batch_labels:

            action_len = env.action_dim
            if env.ld_adaptor:
                action_len = env.ld_adaptor.high_dim
            action_trend = [0] * action_len
            # print(action_trend)

            node_i = 0
            cnt = 0
            for se in env.se_info:
                for key in se.tune_action.keys():
                    # print("[LERC DEBUG] label = ", label)
                    # print("[LERC DEBUG] cnt = ", cnt)
                    self.node_labels_to_action_trend(
                        label[node_i], key, action_trend, cnt, se)
                    # if globalValue.MAX_REWARD <= 0 and key == 'buffer_pool_size':
                    #     action_trend[cnt] = -1
                    cnt += 1
                node_i += 1

            for ce in env.ce_info:
                for key in ce.tune_action.keys():
                    self.node_labels_to_action_trend(
                        label[node_i], key, action_trend, cnt, ce)
                    # if globalValue.MAX_REWARD <= 0 and key == 'buffer_pool_size':
                    #     action_trend[cnt] = -1
                    cnt += 1
                node_i += 1

            batch_trend.append(action_trend)

        return batch_trend

    # 人类先验知识控制器
    # 数据库状态转化为标签
    def status_to_labels(self, obs, node):
        # print("[MERGE DEBUG] lerc status to label obs = {0}".format(obs))
        labelA = labelB = labelC = -1
        data_list = obs
        
        
        # print("xxx data_list = ", data_list)
        # 0 - free_size
        # 1 - LRU_size
        # 3 - flush_size
        # 6 - wait_write_flush
        # 10 - read_pages_per_second
        # 12 - write_pages_per_second
        # labelA:0-写密集，1-读密集
        if data_list[10] < data_list[12] or data_list[3] > 0.5 * data_list[1] or data_list[6] > 0:
            labelA = 0
        else:
            labelA = 1

        # 0 - free_size
        # 11 - create_pages_per_second
        # labelB:0-低负载，1-高负载
        # if (data_list[0] == 0 and data_list[11] > 0) or (data_list[14] < 0 and data_list[15] < 0):
        if ((data_list[0] * 1.0)  < (data_list[1] * 0.8) and node.name == 'ce') or (data_list[0] == 0 and data_list[11] > 0) or (node.delta_q0 < 0 and node.delta_h0 < 0):

            labelB = 1
        else:
            labelB = 0

        # current_ios--->io_capacity
        # 7 - current_ios
        if data_list[7] < 200:
            labelC = 0
        else:
            labelC = 1
        # print(node.name + ' label:' + str(node.labelA) + str(node.labelB) + str(node.labelC))

        return [labelA, labelB, labelC]


class LERCAgent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(LERCAgent, self).__init__(algorithm)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            batch_obs = layers.data(
                name='batch_obs', shape=[self.obs_dim], dtype='float32')
            # batch_init_actions = layers.data(
            #     name='batch_init_actions', shape=[self.act_dim], dtype='float32')
            batch_delta_actions = layers.data(
                name='batch_delta_actions', shape=[self.act_dim], dtype='float32')
            self.lerc_reward_vec = self.alg.predict(
                batch_delta_actions, batch_obs)

        with fluid.program_guard(self.learn_program):

            batch_expert_trends = layers.data(
                name='batch_expert_trends', shape=[self.act_dim], dtype='float32')
            batch_delta_actions = layers.data(
                name='batch_delta_actions', shape=[self.act_dim], dtype='float32')
            batch_obs = layers.data(
                name='batch_obs', shape=[self.obs_dim], dtype='float32')
            batch_rewards = layers.data(
                name='batch_rewards', shape=[], dtype='float32')

            # print(batch_expert_trends)
            # print(layers.shape(batch_expert_trends)[0])

            self.lerc_reward_vec_test = self.alg.learn(
                batch_expert_trends, batch_delta_actions, batch_obs, batch_rewards)

    def predict(self, batch_delta_actions, batch_obs):
        # batch_init_actions = np.expand_dims(batch_init_actions, axis=0)
        batch_delta_actions = np.expand_dims(batch_delta_actions, axis=0)
        batch_obs = np.expand_dims(batch_obs, axis=0)

        feed = {
            'batch_obs': batch_obs,
            'batch_delta_actions': batch_delta_actions,
            # 'batch_init_actions': batch_init_actions
        }
        # print('===obs===:', obs)
        lerc_reward_vec = self.fluid_executor.run(
            self.pred_program, feed=feed,
            fetch_list=[self.lerc_reward_vec])[0]
        # print("self.lerc_reward_vec = ", self.lerc_reward_vec)
        # print("lerc_reward_vec = ", lerc_reward_vec)
        # print(act)
        lerc_reward_vec = np.squeeze(lerc_reward_vec)
        return lerc_reward_vec

    def learn(self, batch_expert_trends, batch_delta_actions, batch_obs, batch_rewards):
        # self.learn_it += 1
        batch_expert_trends = np.expand_dims(batch_expert_trends, axis=0)
        batch_delta_actions = np.expand_dims(batch_delta_actions, axis=0)
        batch_obs = np.expand_dims(batch_obs, axis=0)
        batch_rewards = np.expand_dims(batch_rewards, axis=0)

        # print("batch_expert_trends = ", batch_expert_trends)
        # print("batch_delta_actions = ", batch_delta_actions)
        # print("batch_obs = ", batch_obs)
        # print("batch_rewards = ", batch_rewards)

        feed = {
            'batch_expert_trends': batch_expert_trends,
            'batch_delta_actions': batch_delta_actions,
            'batch_obs': batch_obs,
            'batch_rewards': batch_rewards
        }

        lerc_reward_vec_test = self.fluid_executor.run(
            self.learn_program,
            feed=feed,
            fetch_list=[self.lerc_reward_vec_test])[0]
        # print('critic loss = ', critic_cost[0])

        # self.alg.sync_target()
        # return critic_cost[0], actor_cost[0]

    # def save(self, save_path, mode, program=None):
    #     """Save parameters.
    #     Args:
    #         save_path(str): where to save the parameters.
    #         program(fluid.Program): program that describes the neural network structure. If None, will use self.learn_program.
    #     Raises:
    #         ValueError: if program is None and self.learn_program does not exist.
    #     Example:
    #     .. code-block:: python
    #         agent = AtariAgent()
    #         agent.save('./model.ckpt')
    #     """
    #     # if mode == "train_actor":
    #     #     save_program = self.actor_learn_program
    #     # elif mode == "train_critic":
    #     #     save_program = self.critic_learn_program
    #     if mode == "predict":
    #         save_program = self.pred_program
    #     elif mode == "learn":
    #         save_program = self.learn_program
    #     else:
    #         save_program = self.pred_program
    #     if program is None:
    #         program = save_program
    #     dirname = os.sep.join(save_path.split(os.sep)[:-1])
    #     filename = save_path.split(os.sep)[-1]
    #     fluid.io.save_params(
    #         executor=self.fluid_executor,
    #         dirname=dirname,
    #         main_program=program,
    #         filename=filename)

    # def restore(self, save_path, mode, program=None):
    #     """Restore previously saved parameters.
    #     This method requires a program that describes the network structure.
    #     The save_path argument is typically a value previously passed to ``save_params()``.
    #     Args:
    #         save_path(str): path where parameters were previously saved.
    #         program(fluid.Program): program that describes the neural network structure. If None, will use self.learn_program.
    #     Raises:
    #         ValueError: if program is None and self.learn_program does not exist.
    #     Example:
    #     .. code-block:: python
    #         agent = AtariAgent()
    #         agent.save('./model.ckpt')
    #         agent.restore('./model.ckpt')
    #     """
    #     # if mode == "train_actor":
    #     #     save_program = self.actor_learn_program
    #     # elif mode == "train_critic":
    #         # save_program = self.critic_learn_program
    #     if mode == "predict":
    #         save_program = self.pred_program
    #     elif mode == "learn":
    #         save_program = self.learn_program
    #     else:
    #         save_program = self.pred_program
    #     if program is None:
    #         program = save_program
    #     if type(program) is fluid.compiler.CompiledProgram:
    #         program = program._init_program
    #     dirname = os.sep.join(save_path.split(os.sep)[:-1])
    #     filename = save_path.split(os.sep)[-1]
    #     fluid.io.load_params(
    #         executor=self.fluid_executor,
    #         dirname=dirname,
    #         main_program=program,
    #         filename=filename)


class LERCAlg(Algorithm):
    def __init__(
        self,
        lerc_model,
        lerc_lr,
        use_learn=False,
        batch_size=16,
        act_dim=25,
        obs_dim=28
    ):
        self.lerc_model = lerc_model
        self.lerc_lr = lerc_lr
        self.use_learn = use_learn
        self.batch_size = batch_size
        self.act_dim = act_dim
        self.obs_dim = obs_dim

    def learn(self, batch_expert_trends, batch_delta_actions, batch_obs, batch_rewards):
        lerc_reward_vec = []
        # layers.Print(layers.shape(batch_expert_trends)[0])
        if self.use_learn and layers.shape(batch_expert_trends)[0] >= 0:
            # 计算符号
            # 假设 batch_expert_trends 原本是 int64 类型
            batch_expert_trends_float = layers.cast(
                batch_expert_trends, dtype='float32')
            signs_trends = layers.sign(batch_expert_trends_float)
            signs_actions = layers.sign(batch_delta_actions)

            # 创建结果向量，默认为 -1（符号不匹配）
            # res = layers.fill_constant(shape=[batch_expert_trends.shape[0], 1], dtype='float32', value=-1)
            # res = fluid.layers.fill_constant_batch_size_like(input=batch_expert_trends, value=0, shape=layers.shape(batch_expert_trends), dtype='float32')
            res = layers.fill_constant(
                shape=[self.batch_size, self.act_dim],  # 直接指定形状
                dtype='float32',              # 指定数据类型
                value=1                       # 指定填充的常数值
            )
            # layers.Print(res)

            # 符号相同，设为 1，相反设置为-1
            mask_same_sign = layers.equal(signs_trends, signs_actions)
            mask_diff_sign = layers.logical_not(mask_same_sign)
            mask_same_sign_float = layers.cast(
                mask_same_sign, dtype='float32')  # 将布尔掩码转换为float32
            mask_diff_sign_float = layers.cast(
                mask_diff_sign, dtype='float32') * (-1.0)  # 将布尔掩码转换为float32
            mask_same_sign_float = layers.reshape(mask_same_sign_float, shape=[
                                                  self.batch_size, self.act_dim])
            mask_diff_sign_float = layers.reshape(mask_diff_sign_float, shape=[
                                                  self.batch_size, self.act_dim])

            # res = layers.elementwise_mul(res, mask_same_sign_float, axis=0) + mask_same_sign_float
            res = layers.elementwise_add(
                mask_same_sign_float, mask_diff_sign_float)
            # layers.Print(mask_same_sign_float)
            # layers.Print(mask_diff_sign_float)

            # layers.Print(res)

            # 任一为零，结果为 0

            # 创建一个与这些张量形状相同的0值张量
            zero_trend_tensor = layers.fill_constant(
                shape=[self.batch_size, self.act_dim], dtype='float32', value=0)
            zero_action_tensor = layers.fill_constant(
                shape=[self.batch_size, self.act_dim], dtype='float32', value=0)

            # 现在使用layers.equal进行比较
            mask_zero = layers.logical_or(layers.equal(
                signs_trends, zero_trend_tensor), layers.equal(signs_actions, zero_action_tensor))
            # 假设 mask_zero 是一个布尔型的张量
            mask_not_zero = layers.logical_not(mask_zero)  # 取反得到另一个布尔型张量
            # 将布尔型张量转换为 float32 或 int32 类型，以便用于 elementwise_mul
            mask_not_zero_float = layers.cast(mask_not_zero, dtype='float32')
            mask_not_zero_float = layers.reshape(mask_not_zero_float, shape=[
                                                 self.batch_size, self.act_dim])
            res = layers.elementwise_mul(res, mask_not_zero_float, axis=0)
            # layers.Print(mask_not_zero_float)
            # layers.Print(res)

            input = layers.reshape(layers.concat(input=[
                                   batch_delta_actions, batch_obs], axis=-1), shape=[self.batch_size, self.obs_dim+self.act_dim])
            lerc_reward_vec = self.lerc_model.forward(input)

            new_lerc_reward_vec = layers.elementwise_mul(res, lerc_reward_vec)
            if layers.reduce_sum(layers.abs(res), dim=1, keep_dim=True) != 0:
                mean_lerc_reward_vec = layers.reduce_sum(
                    new_lerc_reward_vec, dim=1, keep_dim=True) / layers.reduce_sum(layers.abs(res), dim=1, keep_dim=True)
            else:
                mean_lerc_reward_vec = layers.reduce_mean(
                    new_lerc_reward_vec, dim=1, keep_dim=True)
            # layers.Print(signs_trends)
            # layers.Print(signs_actions)
            # layers.Print(res)
            # layers.Print(batch_rewards)
            # layers.Print(lerc_reward_vec)
            # layers.Print(new_lerc_reward_vec)
            # layers.Print(mean_lerc_reward_vec)

            # 计算损失并进行反向传播更新模型参数
            loss = layers.reduce_mean(layers.square_error_cost(
                input=mean_lerc_reward_vec, label=batch_rewards))
            optimizer = fluid.optimizer.AdamOptimizer(learning_rate=LERC_LR)
            optimizer.minimize(loss)
            # layers.Print(loss)

            # if ENABLE_DEBUG_INFO:
            #     fluid.layers.Print(batch_expert_trends)
            #     fluid.layers.Print(batch_delta_actions)
            #     fluid.layers.Print(batch_obs)
            return lerc_reward_vec
        else:
            return lerc_reward_vec

    def predict(self, batch_delta_actions, batch_obs):
        batch_delta_actions = batch_delta_actions.astype('float32')
        batch_obs = batch_obs.astype('float32')
        # layers.Print(batch_delta_actions)
        # layers.Print(batch_obs)
        input = layers.reshape(layers.concat(input=[
                               batch_delta_actions, batch_obs], axis=-1), shape=[self.batch_size, self.obs_dim+self.act_dim])
        lerc_reward_vec = self.lerc_model.forward(input)
        # layers.Print(lerc_reward_vec)

        return lerc_reward_vec
