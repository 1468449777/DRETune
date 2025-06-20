import parl
from parl import layers
import numpy as np
from maEnv import datautils
from paddle import fluid
import os
from parl.core.fluid.algorithm import Algorithm
import random
import globalValue
from my_algorithm.RulesController.rule_shap import analyse


SINGLE_NODE_OBS_DIM = 16

ENABLE_DEBUG_INFO = False


# 现在的规则只提供趋势信息
# todo：找到提供幅度信息的规则
class RULES():
    def __init__(self,env):
        self.knobs_range = [
            ["int", 0, 10000, "free_list_len"],          # 0     free_list_len
            ["int", 0, 10000, "lru_len"],               # 1     lru_len
            ["int", 0, 10000, "old_lru_len"],           # 2     old_lru_len
            ["int", 0, 10000, "flush_list_len"],        # 3     flush_list_len
            ["int", 0, 10000, "n_pend_reads"],          # 4     n_pend_reads
            ["int", 0, 10000, "n_pending_flush_lru"],   # 5     n_pending_flush_lru
            ["int", 0, 10000, "n_pending_flush_list"],  # 6     n_pending_flush_list
            ["int", -10000, 10000, "io_cur"],           # 7     io_cur
            ["int", -10000, 10000, "activity_count"],   # 8     activity_count
            ["int", 0, 10000, "n_page_get_delta"],      # 9     n_page_get_delta
            ["double", 0.0, 1.0, "pages_read_rate"],    # 10    pages_read_rate
            ["double", 0.0, 1.0, "pages_created_rate"], # 11    pages_created_rate
            ["double", 0.0, 1.0, "pages_written_rate"], # 12    pages_written_rate
            ["double", 0.0, 1.0, "pages_evicted_rate"]  # 13    pages_evicted_rate
        ]
     
  
        # 状态->标签 在这里注册,同时在下方实现
        self.status_rules_groups = {
            'labelA': [self.ruleA1, self.ruleA2, self.ruleA3],  # 规则组
            'labelB': [self.ruleB1, self.ruleB2, self.ruleB3],  
            'labelC': [self.ruleC1],   
        }  # 用于存储规则组
        self.action_dim = env.dim

    # 根据 name 找到对应的 index
    def get_index_by_name(name,knobs_range):
        for index, knob in enumerate(knobs_range):
            if len(knob) == 4 and knob[3] == name:  # 确保有 name 字段
                return index
        return -1  # 如果未找到，返回 -1

    def get_rule_group(self, group_name):
        """
        获取指定规则组
        :param group_name: 规则组名称
        :return: 规则函数列表
        """
        return self.rules_groups.get(group_name, [])   

    def execute_rule_group(self, group_name, data_list, node):
        """
        执行指定规则组中的规则
        :param group_name: 规则组名称
        :param data_list: 输入数据
        :param node: 节点对象
        :return: 匹配的规则结果
        """
        rules = self.get_rule_group(group_name)
        for rule in rules:
            result = rule(data_list, node)
            if result is not None:  # 如果规则返回了结果，则直接返回
                return result
        return -1  # 如果没有规则匹配，返回默认值 



    def get_labels(self,obs,node):
        """
        根据 status_rules_groups 动态执行规则组并生成标签
        :param obs: 输入数据列表
        :param node: 节点对象
        :return: 标签列表 [labelA, labelB, labelC]
        """
        labels = []
        data_list = obs

        # 遍历 status_rules_groups 中的规则组
        for label_name, rules in self.status_rules_groups.items():
            # 执行规则组并获取结果
            label_result = self.execute_rule_group(label_name, data_list, node)
            labels.append(label_result)

        return labels

    def get_trends(self):
        return self.trends




# 这里是实现的一条条规则，
# todo，使用bert等模型来实现规则的自动生成

# 状态->标签 规则
    def ruleA1(data_list, node):
        if data_list[10] < data_list[12]:
            return 0
        return 1

    def ruleA2(data_list, node):
        if data_list[3] > 0.5 * data_list[1]:
            return 0
        return 1
    
    def ruleA3(data_list, node):
        if data_list[6] > 0:
            return 0
        return 1

    def ruleB1(data_list, node):
        if (data_list[0] * 1.0) < (data_list[1] * 0.8) and node.name == 'ce':
            return 1
        return 0

    def ruleB2(data_list, node):
        if data_list[0] == 0 and data_list[11] > 0:
            return 1
        if node.delta_q0 < 0 and node.delta_h0 < 0:
            return 1
        return 0
    
    def ruleB3(data_list, node):
        if node.delta_q0 < 0 and node.delta_h0 < 0:
            return 1
        return 0    

    def ruleC1(data_list, node):
        if data_list[7] < 200:
            return 0
        return 1

# 标签->动作趋势 规则 







# 以上是实现的一条条规则



# LERC
class LERC():
    def __init__(
        self, 
        obs_dim, 
        action_dim, 
        node_num = 2, 
        use_ld=False, 
        ld_adaptor=None
    ):
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.node_num = node_num
        self.use_ld = use_ld
        self.ld_adaptor = ld_adaptor
        self.rules = RULES()

    

        
    
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

        
        s_total_num = (int)(total_num / globalValue.FAST_LEARN_A_PER_S)
        s_list = np.zeros((s_total_num * self.node_num, int(self.obs_dim / self.node_num)), dtype=float)
        
        # 生成状态参数
        primary_idx = len(globalValue.CONNECT_SE_IP)
        # print(s_total_num, total_num)
        for j in range(int(self.obs_dim / self.node_num)):
            for i in range(s_total_num * self.node_num):
                if j < 14 and self.rules.knobs_range[j][0] == "int":
                    s_list[i][j] = my_rand.randint(self.rules.knobs_range[j][1], self.rules.knobs_range[j][2])
                elif j < 14 and self.rules.knobs_range[j][0] == "double":
                    s_list[i][j] = my_rand.uniform(self.rules.knobs_range[j][1], self.rules.knobs_range[j][2])
            
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
            

        # 用 list comprehension 创建包含 100 个 s 的列表
        new_s_list_t = [new_s_t for _ in range(globalValue.FAST_LEARN_A_PER_S)]
        

        # 使用 np.concatenate 将这些数组沿第一个轴拼接
        new_s_list = np.concatenate(new_s_list_t, axis=0)
        
        
        
        
        # 上一次动作生成
        l_a_list_ld = my_rand.uniform(low=-0.99, high=0.99, size=(total_num, self.action_dim))
        a_list = my_rand.uniform(low=-0.3, high=0.3, size=(total_num, self.action_dim))
        
        # 根据专家知识生成相应动作和奖励
        # 获取 trend

        
        self.get_trend(new_s_list, env)
        # logits test
        self.batch_trends = np.array(self.batch_trends)+1


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

