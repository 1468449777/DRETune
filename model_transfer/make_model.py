
from model_transfer import load_feature
import os
import numpy as np
import globalValue
from my_algorithm.sac_2 import SAC_2
from my_algorithm.sac_model import SACModel
from my_algorithm.agent import SAC2Agent
import json
from hes.low_dim_adaptor import LowDimAdaptor
import pickle
from my_algorithm.replay_memory import ReplayMemory
import time


# h_params of SAC:
H_SAC_ACTOR_LR = 0.0001     # Actor网络的 learning rate
H_SAC_CRITIC_LR = 0.0002    # Critic网络的 learning rate
H_SAC_GAMMA = 0.95          # reward 的衰减因子
H_SAC_TAU = 0.005           # 软更新的系数
H_SAC_ALPHA = 0.2          # 温度参数，决定了熵相对于奖励的相对重要性
H_SAC_ALPHA_LR = 0.0001


BATCH_SIZE = 16


USE_TEST_VEC = True
# TEST_VEC = [69640.0, 6964.0, 13928.0, 6964.0, -0.007617, -0.007617, -0.015234, -0.07617, -0.030468, -0.106638, -0.007617, -0.030468, -0.015234, -0.083787, -0.007617, -0.099021, -0.007617, -0.007617, -0.007617, -0.007617, -0.099021, -0.07617, 31629, 1138, 440, 256, 0, 0, 0, 322, 15462, 158472, 4.11065, 0.0, 308.632, 0.0, 6895, 1159, 447, 700, 0, 0, 0, 0, 57496, 2922070, 13.5206, 4.77073, 0.0, 0.0]
# TEST_VEC = [230200.0, 22906.0, 45895.0, 22920.0, -0.007589, -0.007594, -0.015207, -0.076272, -0.030357, -0.106662, -0.007607, -0.030357, -0.015207, -0.083866, -0.007589, -0.099073, -0.007589, -0.007641, -0.007589, -0.007599, -0.099073, -0.076272, 16194, 11509, 4268, 391, 0, 0, 0, 13, 143, 322, 12.987, 0.0, 0.0, 0.0, 1912, 5924, 2178, 4087, 85, 0, 0, 1452, 52201, 1174365, 556.384, 45.913, 0.0, 0.0, 1332, 6858, 2511, 3, 3, 0, 0, 15077, 0, 7165822, 8.64218e-06, 1.72338e-09, 0.0, 0.0]
TEST_VEC = [24876.0, 5478.0, 11204.0, 5534.0, -0.011648, -0.011767, -0.023824, -0.052896, -0.046594, -0.100136, -0.011993, -0.046594, -0.023824, -0.064664, -0.016924, -0.088488, -0.011648, -0.017283, -0.011648, -0.011831, -0.088488, -0.052896, 12307, 19973, 7352, 1794, 0, 0, 0, 2257, 4661, 107021, 1918.34, 0.0, 2105.1, 0.0, 1760, 6254, 2328, 3662, 38, 0, 0, 1065, 666, 24533, 1338000.0, 190000.0, 0.0, 0.0, 1065, 7125, 2610, 3, 49, 0, 0, 15101, 0, 9113949, 8.6502e-06, 1.72212e-09, 0.0, 0.0]
MODELS_PATH = "./model_transfer/models"
RPMS_PATH = "./model_transfer/rpms"
USE_RPMS_NAME = ["rw"]
SINGLE_NODE_OBS_DIM = 16
NODE_NUM = 3
TOTAL_NUM = 10000

TEST = 123




# 定义函数来读取feature.txt中的数组
def read_feature_file(file_path):
    try:
        # 这里假设数组是以空格或逗号分隔的数字，可以根据实际情况修改
        with open(file_path, 'r') as file:
            # 假设feature.txt里面存储的是空格分隔的浮点数数组
            return np.array([float(x) for x in file.read().split(", ")])
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def get_feature_meta(directory, use_name = None):
    struct_array = {}
    
    # 遍历指定目录下的所有文件夹
    for folder_name in os.listdir(directory):
        # print(folder_name)
        if use_name is not None and folder_name not in use_name:
            continue
        folder_path = os.path.join(directory, folder_name)
        
        if os.path.isdir(folder_path):  # 只处理文件夹
            feature_file_path = os.path.join(folder_path, "feature.txt")
            
            if os.path.isfile(feature_file_path):  # 确保feature.txt存在
                load_vec = read_feature_file(feature_file_path)
                
                
                
                if load_vec is not None:
                    struct_array[folder_name] = {
                        "load_name": folder_name,
                        "load_vec": load_vec,
                    }
    
    return struct_array

# 遍历目录并构建结构体数组
def get_history_load_meta(directory, use_name = None):
    struct_array = {}
    
    # 遍历指定目录下的所有文件夹
    for folder_name in os.listdir(directory):
        if use_name is not None and folder_name not in use_name:
            continue
        folder_path = os.path.join(directory, folder_name)
        
        if os.path.isdir(folder_path):  # 只处理文件夹
            feature_file_path = os.path.join(folder_path, "feature.txt")
            meta_json_file_path = os.path.join(folder_path, "meta.json")
            
            if os.path.isfile(feature_file_path):  # 确保feature.txt存在
                load_vec = read_feature_file(feature_file_path)
                load_vec = normalize_state_vec(load_vec, mode=2)
                with open(meta_json_file_path, 'r') as file:
                    meta_json = json.load(file)
                
                
                
                if load_vec is not None:
                    struct_array[folder_name] = {
                        "load_name": folder_name,
                        "load_vec": load_vec,
                        "model_meta": meta_json
                    }
    
    return struct_array

# # 定义cal_his_cur_similarity函数
# def cal_his_cur_similarity(cur_feature, struct_array):
#     sim_list = []
    
#     all_load_vecs = [cur_feature]
#     all_load_names = ["cur"]
#     # 遍历struct_array中的每个结构体，计算相似度
#     for (key, struct) in struct_array.items():
#         # print("key = {0}, struct = {1}".format(key, struct))
#         load_vec = struct['load_vec']
#         all_load_vecs.append(load_vec)
#         all_load_names.append(struct['load_name'])
    
#     # 将向量转化为 NumPy 数组
#     all_vectors = np.array(all_load_vecs)
    
#     # 计算每个特征的均值和标准差
#     means = np.mean(all_vectors, axis=0)  # 计算每列（特征）的均值
#     stds = np.std(all_vectors, axis=0)    # 计算每列（特征）的标准差
    
#     # 处理标准差为零的情况，避免除以零
#     stds = np.where(stds == 0, 1.0, stds)  # 将标准差为零的地方替换为1，避免除以零
    
#     # 标准化数据
#     normalized_vectors = (all_vectors - means) / stds
        
#     for idx, load_vec in enumerate(normalized_vectors):
#         if idx == 0:    continue
        
#         # 计算cur_feature和load_vec之间的相似度
#         similarity = load_feature.standardize_and_compute_similarity(normalized_vectors[0], load_vec)
        
#         # 将相似度存储在sim_list中
#         sim_list.append({
#             "load_name": all_load_names[idx],
#             "similarity": abs(similarity),
#         })
    
#     # 计算相似度的总和
#     total_similarity = sum([sim['similarity'] for sim in sim_list])
    
    
#     # 如果总和不为零，调整所有相似度，使总和为1
#     if total_similarity != 0:
#         for sim in sim_list:
#             # 更新结构体中的sim字段
#             struct_array[sim['load_name']]['sim'] = sim['similarity'] / total_similarity
#     return struct_array  # 返回更新后的结构体数组


# 定义cal_his_cur_similarity函数
def cal_his_cur_similarity(cur_feature, struct_array):
    sim_list = []
    
    all_load_vecs = [cur_feature]
    all_load_names = ["cur"]
    # 遍历struct_array中的每个结构体，计算相似度
    for (key, struct) in struct_array.items():
        # print("key = {0}, struct = {1}".format(key, struct))
        load_vec = struct['load_vec']
        all_load_vecs.append(load_vec)
        all_load_names.append(struct['load_name'])
    
    # 将向量转化为 NumPy 数组
    all_vectors = np.array(all_load_vecs)
    
    # # 计算每个特征的均值和标准差
    # means = np.mean(all_vectors, axis=0)  # 计算每列（特征）的均值
    # stds = np.std(all_vectors, axis=0)    # 计算每列（特征）的标准差
    
    # # 处理标准差为零的情况，避免除以零
    # stds = np.where(stds == 0, 1.0, stds)  # 将标准差为零的地方替换为1，避免除以零
    
    # 标准化数据
    # normalized_vectors = (all_vectors - means) / stds
    normalized_vectors = all_vectors
        
    for idx, load_vec in enumerate(normalized_vectors):
        if idx == 0:    continue
        
        # 计算cur_feature和load_vec之间的相似度
        similarity = load_feature.standardize_and_compute_similarity(normalized_vectors[0], load_vec)
        
        # 将相似度存储在sim_list中
        sim_list.append({
            "load_name": all_load_names[idx],
            "similarity": abs(similarity),
            "raw_sim": similarity,
        })
    
    # 计算相似度的总和
    total_similarity = sum([sim['similarity'] for sim in sim_list])
    
    
    # 如果总和不为零，调整所有相似度，使总和为1
    if total_similarity != 0:
        for sim in sim_list:
            # 更新结构体中的sim字段
            struct_array[sim['load_name']]['sim'] = sim['similarity'] / total_similarity
            struct_array[sim['load_name']]['raw_sim'] = sim['raw_sim']
    return struct_array  # 返回更新后的结构体数组





def normalize_state_vec(nodes_state, mode = 2):
    
    crud_range = [
        ["int", 0, 1],          # 0     free_list_len
        ["int", 0, 1],          # 1     lru_len
        ["int", 0, 1],          # 2     old_lru_len
        ["int", 0, 1],          # 3     flush_list_l
    ]
    
    knobs_range = [
        ["int", 0, 100000],          # 0     free_list_len
        ["int", 0, 100000],          # 1     lru_len
        ["int", 0, 100000],          # 2     old_lru_len
        ["int", 0, 100000],          # 3     flush_list_len
        ["int", 0, 100000],          # 4     n_pend_reads
        ["int", 0, 100000],          # 5     n_pending_flush_lru
        ["int", 0, 100000],          # 6     n_pending_flush_list
        ["int", -100000, 100000],     # 7     io_cur
        ["int", -100000, 100000],     # 8     activity_count
        ["int", 0, 100000],          # 9     n_page_get_delta
        ["double", 0.0, 1000.0],       # 10    pages_read_rate
        ["double", 0.0, 1000.0],       # 11    pages_created_rate
        ["double", 0.0, 1000.0],       # 12    pages_written_rate
        ["double", 0.0, 1000.0]        # 13    pages_evicted_rate
        # you can add more ranges here
    ]
    
    if mode == 1:
    
        # print("before normalize: {0}".format(nodes_state))
        node_obs_dim = 14
        for idx, s in enumerate(nodes_state):
            mod = idx % node_obs_dim
            nodes_state[idx] = (float)(nodes_state[idx] - knobs_range[mod][1]) / (float)(knobs_range[mod][2] - knobs_range[mod][1])
        
        # print("after  normalize: {0}".format(nodes_state))
        return nodes_state
    elif mode == 2:
        part_node_state = nodes_state[22:]
        # print("before normalize(after extract): {0}".format(part_node_state))
        # print("before normalize(after extract) len = : {0}".format(len(part_node_state)))
        node_obs_dim = 14
        for idx, s in enumerate(part_node_state):
            mod = idx % node_obs_dim
            part_node_state[idx] = (float)(part_node_state[idx] - knobs_range[mod][1]) / (float)(knobs_range[mod][2] - knobs_range[mod][1])
        
        # print("after  normalize(after extract): {0}".format(part_node_state))
        
        for idx, s in enumerate(nodes_state):
            if idx >= 22:
                nodes_state[idx] = part_node_state[idx-22]
        # print("after  normalize(all   states ): {0}".format(nodes_state))
        
        
        part_node_state = nodes_state[0:4]
        # print("before normalize(after extract): {0}".format(part_node_state))
        # print("before normalize(after extract) len = : {0}".format(len(part_node_state)))
        node_obs_dim = 4
        for idx, s in enumerate(part_node_state):
            mod = idx % node_obs_dim
            part_node_state[idx] = (float)(part_node_state[idx] - crud_range[mod][1]) / (float)(crud_range[mod][2] - crud_range[mod][1])
        
        # print("after  normalize(after extract): {0}".format(part_node_state))
        
        for idx, s in enumerate(nodes_state):
            if idx < 4:
                nodes_state[idx] = part_node_state[idx]
        # print("after  normalize(all   states ): {0}".format(nodes_state))
        
        
        return nodes_state
        


def load_his_models(history_meta, his_agent_dict, his_ld_dict, his_agent_double_dict, his_ld_double_dict):
    max_action = 0.99999999
    
    
    
    
    
    for (key, struct) in history_meta.items():
        
        model =  SACModel(struct['model_meta']['low_act_dim'], struct['model_meta']['obs_dim'])
        algorithm = SAC_2(
            actor=model.actor_model,
            critic=model.critic_model,
            alpha_model=model.alpha_model,
            reward_model=model.reward_model,
            max_action=max_action,
            alpha=H_SAC_ALPHA,
            gamma=H_SAC_GAMMA,
            tau=H_SAC_TAU,
            actor_lr=H_SAC_ACTOR_LR,
            critic_lr=H_SAC_CRITIC_LR,
            alpha_lr=H_SAC_ALPHA_LR,
            batch_size=BATCH_SIZE,
            states_model=model.states_model,
            
        )
        
        # agent = SAC2Agent(algorithm, struct['model_meta']['obs_dim'], struct['model_meta']['low_act_dim'], struct['model_meta']['low_act_dim'], struct['model_meta']['high_act_dim'])
        agent = SAC2Agent(algorithm, struct['model_meta']['obs_dim'], struct['model_meta']['low_act_dim'], struct['model_meta']['low_act_dim'], struct['model_meta']['high_act_dim'])
        model_path = os.path.join(MODELS_PATH, struct['load_name'], "actor.ckpt")
        # 导入已有模型
        agent.restore(model_path, mode='predict')
        
        
        model_double =  SACModel(struct['model_meta']['low_act_dim'], struct['model_meta']['obs_dim'], True)
        algorithm_double = SAC_2(
            actor=model_double.actor_model,
            critic=model_double.critic_model,
            alpha_model=model_double.alpha_model,
            reward_model=model_double.reward_model,
            max_action=max_action,
            alpha=H_SAC_ALPHA,
            gamma=H_SAC_GAMMA,
            tau=H_SAC_TAU,
            actor_lr=H_SAC_ACTOR_LR,
            critic_lr=H_SAC_CRITIC_LR,
            alpha_lr=H_SAC_ALPHA_LR,
            batch_size=BATCH_SIZE,
            states_model=model_double.states_model,
            is_double=True,
        )
        agent_double = SAC2Agent(algorithm_double, struct['model_meta']['obs_dim'], struct['model_meta']['low_act_dim'], struct['model_meta']['low_act_dim'], struct['model_meta']['high_act_dim'])
        model_double_path = os.path.join(MODELS_PATH, struct['load_name'], "double_actor.ckpt")
        # 导入已有模型
        agent_double.restore(model_double_path, mode='predict')
        
        
        
        
        
        ld_adaptor = LowDimAdaptor(None, struct['model_meta']['low_act_dim'], struct['model_meta']['seed'], struct['model_meta']['high_act_dim'])
        # if USE_DOUBLE_NETWORK:
        #     env.ld_adaptor_double = LowDimAdaptor(env, globalValue.HES_LOW_ACT_DIM_DOUBLE, hes_low_seed+globalValue.DOUBLE_SEED_DELTA)
        #     # TODO 对生成做判断
        #     env.ld_adaptor_double.double_network_correct(env.ld_adaptor.A)
        #     print("===============================================================")
        #     env.ld_adaptor_double.double_network_correct(env.ld_adaptor.A)
        
        ld_adaptor_double = LowDimAdaptor(None, struct['model_meta']['low_act_dim'], struct['model_meta']['seed'], struct['model_meta']['high_act_dim'])
        ld_adaptor_double.double_network_correct(ld_adaptor.A)
        ld_adaptor_double.double_network_correct(ld_adaptor.A)
        
        
        
        
        # test_obs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        # test_last_action = [-0.9393939393939394, -0.28888888888888886, -0.8, 0.0, -0.02564102564102566, -0.36, -0.6, -1.0, 1.0, 0.0, 0.5555555555555554, -1.0, -0.7142857142857143, -0.9419419419419419, -1.0, 0.75, -0.9797979797979798, -0.9393939393939394, -0.28888888888888886, -0.8, -0.0010256410256410664, 0.5555555555555554, -1.0, 0.75, -0.9419419419419419, -0.9393939393939394, -0.28888888888888886, -0.8, -0.0010256410256410664, 0.5555555555555554, -1.0, 0.75, -0.9419419419419419]
        # test_action_pred = agent.predict(test_obs.astype('float32'), np.array([test_last_action]).astype('float32'))
        
        # print("[NOTION] test_action_pred = {0}".format(test_action_pred))
        # test_action_pred_hd = ld_adaptor.transform(test_action_pred)
        # print("[NOTION] test_action_pred_hd = {0}".format(test_action_pred_hd))
        
        
        
        
        # action_trend_1 = agent.get_trend(test_obs.astype('float32'), np.array([test_last_action]).astype('float32'))-1.0
        # action_trend_2 = agent_double.get_trend(test_obs.astype('float32'), np.array([test_last_action]).astype('float32'))-1.0
        # action_ampl_1 = agent.get_ampl(test_obs.astype('float32'), np.array([test_last_action]).astype('float32'))
        # action_ampl_2 = agent_double.get_ampl(test_obs.astype('float32'), np.array([test_last_action]).astype('float32'))
        # # action_1 = np.multiply(action_trend_1, action_ampl_t)
        # # action_2 = np.multiply(action_trend_2, action_ampl_t)
        # action_trend = action_trend_1 * globalValue.DOUBLE_RATIO + action_trend_2 * (1-globalValue.DOUBLE_RATIO) 
        # action_ampl = action_ampl_1 * globalValue.DOUBLE_RATIO + action_ampl_2 * (1-globalValue.DOUBLE_RATIO)
        # action_double = action_trend * action_ampl
        # action_double = test_action_pred_hd
        # # print("[NOTION] action_double = {0}".format(action_double))
        
        
        
        # t_act_dim = 20
        # t_state_dim = 48
        # t_ld = 20
        # t_hd = 33
        # model =  SACModel(t_act_dim, t_state_dim)
        # algorithm = SAC_2(
        #     actor=model.actor_model,
        #     critic=model.critic_model,
        #     alpha_model=model.alpha_model,
        #     reward_model=model.reward_model,
        #     max_action=max_action,
        #     alpha=H_SAC_ALPHA,
        #     gamma=H_SAC_GAMMA,
        #     tau=H_SAC_TAU,
        #     actor_lr=H_SAC_ACTOR_LR,
        #     critic_lr=H_SAC_CRITIC_LR,
        #     alpha_lr=H_SAC_ALPHA_LR,
        #     batch_size=BATCH_SIZE,
        #     states_model=model.states_model,
            
        # )
        # model_path = os.path.join(MODELS_PATH, struct['load_name'], "actor.ckpt")
        # model_path = "./test_ttt.ckpt"
        # model_path = "test_newest_predict_sac_2.ckpt"
        # agent = SAC2Agent(algorithm, t_state_dim, t_act_dim, t_ld, t_hd)
        
        # agent.save("./test_ttt.ckpt", mode='learn')
        
        # t_act_dim = 20
        # t_state_dim = 48
        # t_ld = 20
        # t_hd = 33
        # model2 =  SACModel(t_act_dim, t_state_dim)
        # algorithm2 = SAC_2(
        #     actor=model2.actor_model,
        #     critic=model2.critic_model,
        #     alpha_model=model2.alpha_model,
        #     reward_model=model2.reward_model,
        #     max_action=max_action,
        #     alpha=H_SAC_ALPHA,
        #     gamma=H_SAC_GAMMA,
        #     tau=H_SAC_TAU,
        #     actor_lr=H_SAC_ACTOR_LR,
        #     critic_lr=H_SAC_CRITIC_LR,
        #     alpha_lr=H_SAC_ALPHA_LR,
        #     batch_size=BATCH_SIZE,
        #     states_model=model2.states_model,
            
        # )
        # agent2 = SAC2Agent(algorithm2, t_state_dim, t_act_dim, t_ld, t_hd)
        
        
        # agent2.restore(model_path, mode='predict')
        
        
        
        
        
        his_agent_dict[struct['load_name']] = agent
        his_agent_double_dict[struct['load_name']] = agent_double
        his_ld_dict[struct['load_name']] = ld_adaptor
        his_ld_double_dict[struct['load_name']] = ld_adaptor_double
    return his_agent_dict, his_ld_dict, his_agent_double_dict, his_ld_double_dict



def generate_transfer_exp(total_num, rpm, transfer_info):  
        
    my_rand = np.random.RandomState(globalValue.FAST_LEARN_GEN_SEED)
    
    s_list=[]
    a_list=[]
    l_a_list=[]
    r_list = []
    d_list = np.zeros(total_num, dtype=float)
    n_s_list = []
    
    
    obs_dim = transfer_info['obs_dim']
    node_num = NODE_NUM
    action_dim = transfer_info['ld']
    high_action_dim = transfer_info['hd']
    
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
    
    
    
    
    s_total_num = (int)(total_num)
    s_list = np.zeros((s_total_num * node_num, int(obs_dim / node_num)), dtype=float)
    
    # 生成状态参数
    primary_idx = len(globalValue.CONNECT_SE_IP)
    print(s_total_num, total_num)
    for j in range(int(obs_dim / node_num)):
        for i in range(s_total_num * node_num):
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
    node_s = np.array_split(s_list, node_num)
    
    # print("[S DEBUG] node_s = ", node_s)
    
    
    new_s_t = np.concatenate(node_s, axis=1)
    
    
    
    # print("[S DEBUG] new_s_t = ", new_s_t)
    
    # primary delta_q0
    if not globalValue.USE_PROXY_SQL:
        for i in range(new_s_t.shape[0]):
            qps_idx = SINGLE_NODE_OBS_DIM * (len(globalValue.CONNECT_SE_IP)) + 14
            new_s_t[i][qps_idx] = my_rand.uniform(-1.0, 1.0)
    
    
    

    
    # print("new_s_t = ", new_s_t.shape)
    # print("[FL GEN DEBUG] new_s_list shape = ", new_s_list.shape)
    # new_s_list = np.tile(new_s_list_t, (100, 1))  # 沿第一个维度重复 100 次
    
    # 用 list comprehension 创建包含 100 个 s 的列表
    new_s_list_t = [new_s_t for _ in range(1)]
    

    # 使用 np.concatenate 将这些数组沿第一个轴拼接
    new_s_list = np.concatenate(new_s_list_t, axis=0)
    
    
    
    
    # 上一次动作生成
    l_a_list_hd = my_rand.uniform(low=-0.99, high=0.99, size=(total_num, high_action_dim))
    a_list = my_rand.uniform(low=-0.3, high=0.3, size=(total_num, action_dim))
    a_list_double = my_rand.uniform(low=-0.3, high=0.3, size=(total_num, action_dim))
    
    # 根据专家知识生成相应动作和奖励
    # 获取 trend
    # print("new_s_list = ", new_s_list.shape)
    
    # get_trend(new_s_list, env)
    # logits test
    # batch_trends = np.array(batch_trends)+1
    # print(batch_trends)

    # print("new_s_list = ", new_s_list.shape)
    actual_trend_list = []
    
    a_cnt = 0
    start_time = time.time()
    
    # agent = transfer_info['his_agent_dict']['ro']
    # test_obs = np.array(new_s_list).astype('float32')
    # test_last_action = np.array(l_a_list_hd).astype('float32')
    # action_trend_1 = agent.get_trend(test_obs, test_last_action)-1.0
    # action_trend_2 = agent_double.get_trend(test_obs.astype('float32'), np.array([test_last_action]).astype('float32'))-1.0
    # action_ampl_1 = agent.get_ampl(test_obs.astype('float32'), np.array([test_last_action]).astype('float32'))

    
    
    
    for i in range((int)(total_num/len((list)(transfer_info['his_agent_dict'].keys())))):
        # print("i = ", i)
        for (key, agent) in transfer_info['his_agent_dict'].items():
            agent_double = transfer_info['his_agent_double_dict'][key]
            ld_adaptor = transfer_info['his_ld_dict'][key]
            ld_adaptor_double = transfer_info['his_ld_double_dict'][key]
            
            test_obs = np.array(new_s_list[i])
            # test_obs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            test_last_action = l_a_list_hd[i]
            # test_last_action = [-0.9393939393939394, -0.28888888888888886, -0.8, 0.0, -0.02564102564102566, -0.36, -0.6, -1.0, 1.0, 0.0, 0.5555555555555554, -1.0, -0.7142857142857143, -0.9419419419419419, -1.0, 0.75, -0.9797979797979798, -0.9393939393939394, -0.28888888888888886, -0.8, -0.0010256410256410664, 0.5555555555555554, -1.0, 0.75, -0.9419419419419419, -0.9393939393939394, -0.28888888888888886, -0.8, -0.0010256410256410664, 0.5555555555555554, -1.0, 0.75, -0.9419419419419419]
            
            # print("[TRANFER_DATA NOTE] test_obs.shape = ", test_obs.shape)
            # print("[TRANFER_DATA NOTE] np.array([test_last_action]).shape = ", np.array([test_last_action]).shape)
            test_action_pred = agent.predict(test_obs.astype('float32'), np.array([test_last_action]).astype('float32'))
            
            # print("[NOTION] test_action_pred = {0}".format(test_action_pred))
            # test_action_pred_hd = ld_adaptor.transform(test_action_pred)
            # print("[NOTION] test_action_pred_hd = {0}".format(test_action_pred_hd))
            
            
            
            
            action_trend_1 = agent.get_trend(test_obs.astype('float32'), np.array([test_last_action]).astype('float32'))-1.0
            action_trend_2 = agent_double.get_trend(test_obs.astype('float32'), np.array([test_last_action]).astype('float32'))-1.0
            action_ampl_1 = agent.get_ampl(test_obs.astype('float32'), np.array([test_last_action]).astype('float32'))
            # action_ampl_2 = agent_double.get_ampl(test_obs.astype('float32'), np.array([test_last_action]).astype('float32'))
            action_1 = np.multiply(action_trend_1, action_ampl_1)
            # action_2 = np.multiply(action_trend_2, action_ampl_t)
            # action_trend = action_trend_1 * globalValue.DOUBLE_RATIO + action_trend_2 * (1-globalValue.DOUBLE_RATIO) 
            # action_ampl = action_ampl_1 * globalValue.DOUBLE_RATIO + action_ampl_2 * (1-globalValue.DOUBLE_RATIO)
            # action_double = action_trend * action_ampl
            # action_double = test_action_pred
            # print("[NOTION] action_double = {0}".format(action_double))
            a_list[a_cnt] = action_1
            a_list_double[a_cnt] = action_trend_2
            a_cnt += 1
            r = transfer_info['meta'][key]['sim'] * 0.1
            # print("[NOTION] r = {0}".format(r))
            r_list.append(r)
            
    end_time = time.time()
    print("[TIME] predicr time = ", end_time - start_time)     

        
        
        
        
    #     r_list.append(r)
        
        
    # 下一个状态无用，随便生成
    n_s_list = np.zeros((total_num, obs_dim), dtype=float)
    
    r_list = np.array(r_list)
    # a_list = np.array(a_list)
    l_a_list = np.array(l_a_list_hd)
    # trends_list = np.array(batch_trends)
    
    print("\n======================[FAST LEARN][GEN]======================")
    print("new_s_list shape = ", new_s_list.shape)
    print("a_list shape = ", a_list.shape)
    print("r_list shape = ", r_list.shape)
    print("n_s_list shape = ", n_s_list.shape)
    print("d_list shape = ", d_list.shape)
    print("l_a_list shape = ", l_a_list.shape)
    # print("trends_list shape = ", trends_list.shape)
    print("======================[FAST LEARN][END]======================\n")
    
    
    for i in range(total_num):
        rpm.append((new_s_list[i], a_list[i], r_list[i], n_s_list[i], d_list[i], l_a_list[i]))
    
    # batch_obs, batch_action, batch_reward, batch_next_obs, batch_done
    return new_s_list, a_list, r_list, n_s_list, d_list, l_a_list 



def generate_transfer_exp_use_model(total_num, rpm, rpm_double, transfer_info):   
       

   
    my_rand = np.random.RandomState(globalValue.FAST_LEARN_GEN_SEED)
    
    s_list=[]
    a_list=[]
    l_a_list=[]
    r_list = []
    
    n_s_list = []
    
    
    obs_dim = transfer_info['obs_dim']
    node_num = transfer_info['node_num']
    action_dim = transfer_info['ld']
    high_action_dim = transfer_info['hd']
    
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
    
    
    
    
    s_total_num = (int)(total_num)
    s_list = np.zeros((s_total_num * node_num, int(obs_dim / node_num)), dtype=float)
    
    # 生成状态参数
    primary_idx = len(globalValue.CONNECT_SE_IP)
    print(s_total_num, total_num)
    for j in range(int(obs_dim / node_num)):
        for i in range(s_total_num * node_num):
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
    node_s = np.array_split(s_list, node_num)
    
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
    new_s_list_t = [new_s_t for _ in range(1)]
    

    # 使用 np.concatenate 将这些数组沿第一个轴拼接
    new_s_list = np.concatenate(new_s_list_t, axis=0)
    
    
    init_delta = 0.2
    print("[TRANSFER_DATE DEBUG] transfer_info['init_state'] = ", transfer_info['init_state'])
    print("[TRANSFER_DATE DEBUG] new_s_list.shape = ", new_s_list.shape)
    init_state_num = int(total_num * transfer_info['init_ratio'])
    if init_state_num > 0:
        new_s_list = np.concatenate((new_s_list, np.array([transfer_info['init_state']])), axis=0)
        for x in range(init_state_num - 1):
            tmp_state = transfer_info['init_state']
            
            for y in range(obs_dim):
                p_state = my_rand.uniform(0.0, 1.0)
                if p_state < init_delta:
                    mod_num = y % obs_dim
                    if mod_num < 14 and knobs_range[mod_num][0] == "int":

                        tmp_state[y] = round(tmp_state[y]*my_rand.uniform(1-init_delta, 1+init_delta))
                    elif mod_num < 14 and knobs_range[mod_num][0] == "double":
                        tmp_state[y] = tmp_state[y]*my_rand.uniform(1-init_delta, 1+init_delta)
                    # delta_q0
                    elif mod_num == 14:
                        if globalValue.USE_PROXY_SQL:
                            tmp_state[y] = my_rand.uniform(-1.0, 1.0)
                        else:
                            tmp_state[y] = 0.0
                        # delta_h0
                    elif mod_num == 15:
                        tmp_state[y] = my_rand.uniform(-0.5, 0.5)
            
            new_s_list = np.concatenate((new_s_list, np.array([tmp_state])), axis=0)
    print("[TRANSFER_DATA DEBUG] new_s_list.shape = ", new_s_list.shape)
    print("[S DEBUG] new_s_list = ", new_s_list)
    
    
    
    
    
    
    
    
    
    
    # 上一次动作生成
    l_a_list_hd = my_rand.uniform(low=-0.99, high=0.99, size=(total_num+init_state_num, high_action_dim))
    a_list = my_rand.uniform(low=-0.3, high=0.3, size=(total_num+init_state_num, action_dim))
    a_list_double = my_rand.uniform(low=-0.3, high=0.3, size=(total_num+init_state_num, action_dim))
    
    d_list = np.zeros(total_num+init_state_num, dtype=float)
    # 根据专家知识生成相应动作和奖励
    # 获取 trend
    # print("new_s_list = ", new_s_list.shape)
    
    # get_trend(new_s_list, env)
    # logits test
    # batch_trends = np.array(batch_trends)+1
    # print(batch_trends)

    # print("new_s_list = ", new_s_list.shape)
    actual_trend_list = []
    
    a_cnt = 0
    start_time = time.time()
    agent = transfer_info['his_agent_dict']['default']
    agent_double = transfer_info['his_agent_double_dict']['default']
    # ld_adaptor = transfer_info['his_ld_dict']['default']
    
    for i in range((int)(total_num/len((list)(transfer_info['his_agent_dict'].keys())))+init_state_num):
        # print("i = ", i)
            
            
        test_obs = np.array(new_s_list[i])
        # test_obs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        test_last_action = l_a_list_hd[i]
        # test_last_action = [-0.9393939393939394, -0.28888888888888886, -0.8, 0.0, -0.02564102564102566, -0.36, -0.6, -1.0, 1.0, 0.0, 0.5555555555555554, -1.0, -0.7142857142857143, -0.9419419419419419, -1.0, 0.75, -0.9797979797979798, -0.9393939393939394, -0.28888888888888886, -0.8, -0.0010256410256410664, 0.5555555555555554, -1.0, 0.75, -0.9419419419419419, -0.9393939393939394, -0.28888888888888886, -0.8, -0.0010256410256410664, 0.5555555555555554, -1.0, 0.75, -0.9419419419419419]
        
        # print("[TRANFER_DATA NOTE] test_obs.shape = ", test_obs.shape)
        # print("[TRANFER_DATA NOTE] np.array([test_last_action]).shape = ", np.array([test_last_action]).shape)
        # test_action_pred = agent.predict(test_obs.astype('float32'), np.array([test_last_action]).astype('float32'))
        
        # print("[NOTION] test_action_pred = {0}".format(test_action_pred))
        # test_action_pred_hd = ld_adaptor.transform(test_action_pred)
        # print("[NOTION] test_action_pred_hd = {0}".format(test_action_pred_hd))
        
        
        
        
        action_trend_1 = agent.get_trend(test_obs.astype('float32'), np.array([test_last_action]).astype('float32'))-1.0
        action_trend_2 = agent_double.get_trend(test_obs.astype('float32'), np.array([test_last_action]).astype('float32'))-1.0
        action_ampl_1 = agent.get_ampl(test_obs.astype('float32'), np.array([test_last_action]).astype('float32'))
        # action_ampl_2 = agent_double.get_ampl(test_obs.astype('float32'), np.array([test_last_action]).astype('float32'))
        action_1 = np.multiply(action_trend_1, action_ampl_1)
        # action_2 = np.multiply(action_trend_2, action_ampl_t)
        # action_trend = action_trend_1 * globalValue.DOUBLE_RATIO + action_trend_2 * (1-globalValue.DOUBLE_RATIO) 
        # action_ampl = action_ampl_1 * globalValue.DOUBLE_RATIO + action_ampl_2 * (1-globalValue.DOUBLE_RATIO)
        # action_double = action_trend * action_ampl
        # action_double = test_action_pred
        # print("[NOTION] action_double = {0}".format(action_double))
        a_list[a_cnt] = action_1
        a_list_double[a_cnt] = action_trend_2
        a_cnt += 1
        r = 1
        # print("[NOTION] r = {0}".format(r))
        r_list.append(r)

    end_time = time.time()
    print("[TIME] predict time = ", end_time - start_time)     
            
    

        
        
        
        
    #     r_list.append(r)
        
        
    # 下一个状态无用，随便生成
    n_s_list = np.zeros((total_num+init_state_num, obs_dim), dtype=float)
    
    r_list = np.array(r_list)
    # a_list = np.array(a_list)
    l_a_list = np.array(l_a_list_hd)
    # trends_list = np.array(batch_trends)
    
    print("\n======================[FAST LEARN][GEN]======================")
    print("new_s_list shape = ", new_s_list.shape)
    print("a_list shape = ", a_list.shape)
    print("r_list shape = ", r_list.shape)
    print("n_s_list shape = ", n_s_list.shape)
    print("d_list shape = ", d_list.shape)
    print("l_a_list shape = ", l_a_list.shape)
    # print("trends_list shape = ", trends_list.shape)
    print("======================[FAST LEARN][END]======================\n")
    
    
    for i in range(total_num+init_state_num):
        rpm.append((new_s_list[i], a_list[i], r_list[i], n_s_list[i], d_list[i], l_a_list[i]))
        rpm_double.append((new_s_list[i], a_list_double[i], r_list[i], n_s_list[i], d_list[i], l_a_list[i]))
    
    # batch_obs, batch_action, batch_reward, batch_next_obs, batch_done
    return new_s_list, a_list, r_list, n_s_list, d_list, l_a_list 




# 未调用
def make_model():
    if USE_TEST_VEC:
        cur_load_feature = TEST_VEC
    else:
        cur_load_feature = load_feature.make_load_feature_vec()
    
    # 读取并计算历史负载与当前负载相似度
    history_meta = get_history_load_meta(MODELS_PATH)
    history_meta = cal_his_cur_similarity(cur_load_feature, history_meta)
    print("[NOTION] history_meta = {0}".format(history_meta))
    
    his_agent_dict = {}
    his_agent_double_dict = {}
    his_ld_dict = {}
    his_ld_double_dict = {}
    load_his_models(history_meta, his_agent_dict, his_ld_dict, his_agent_double_dict, his_ld_double_dict)
    

    tmp_key = list(history_meta.keys())[0]
    rpm = ReplayMemory(10000000, history_meta[tmp_key]['model_meta']['low_act_dim'], history_meta[tmp_key]['model_meta']['obs_dim'], True, history_meta[tmp_key]['model_meta']['low_act_dim'], history_meta[tmp_key]['model_meta']['high_act_dim'])
    transfer_info = {
        "meta": history_meta,
        "his_agent_dict": his_agent_dict,
        "his_agent_double_dict": his_agent_dict,
        "his_ld_dict": his_ld_dict,
        "his_ld_double_dict": his_ld_double_dict,
        "ld": history_meta[tmp_key]['model_meta']['low_act_dim'],
        "hd": history_meta[tmp_key]['model_meta']['high_act_dim'],
        "obs_dim": history_meta[tmp_key]['model_meta']['obs_dim']
    }
    print(transfer_info)
    
    generate_transfer_exp(TOTAL_NUM, rpm, transfer_info)
    f_new = open("./model_transfer/transfer_rpm.txt", "wb")    
    pickle.dump(rpm, f_new)
    f_new.close()
    # print(TEST)
    # TEST = 456
    # print(TEST)
    
    # print(his_agent_dict)
    
    
def cal_total_sim(history_meta):
    square_sum = 0
    load_num = 0
    load_name = []
    raw_sim_list = []
    for key, meta in history_meta.items():
        load_num += 1
        load_name.append(key)
        raw_sim_list.append(meta['raw_sim'])
        square_sum += meta['raw_sim'] * meta['raw_sim']
    total_sim = square_sum / load_num
    print("[NOTION] load_name = {0}, total_sim = {1}, sim_list = {2}".format(load_name, total_sim, raw_sim_list))
    
    
    # 未调用
def make_model_from_rpm():
    if USE_TEST_VEC:
        cur_load_feature = TEST_VEC
    else:
        cur_load_feature = load_feature.make_load_feature_vec()
    
    cur_load_feature = normalize_state_vec(cur_load_feature, mode=2)
    
    # 读取并计算历史负载与当前负载相似度
    history_meta = get_history_load_meta(RPMS_PATH, USE_RPMS_NAME)
    history_meta = cal_his_cur_similarity(cur_load_feature, history_meta)
    # print("[NOTION] history_meta = {0}".format(history_meta))
    cal_total_sim(history_meta)
    
    
    tmp_key = list(history_meta.keys())[0]
    cur_rpm = ReplayMemory(10000000, history_meta[tmp_key]['model_meta']['low_act_dim'], history_meta[tmp_key]['model_meta']['obs_dim'], True, history_meta[tmp_key]['model_meta']['low_act_dim'], history_meta[tmp_key]['model_meta']['high_act_dim'])
    cur_rpm_double = ReplayMemory(10000000, history_meta[tmp_key]['model_meta']['low_act_dim'], history_meta[tmp_key]['model_meta']['obs_dim'], True, history_meta[tmp_key]['model_meta']['low_act_dim'], history_meta[tmp_key]['model_meta']['high_act_dim'])

    
    for name, meta in history_meta.items():
        rpm_file_name = os.path.join(RPMS_PATH, name, "rpm.txt")
        rpm_double_file_name = os.path.join(RPMS_PATH, name, "rpm_double.txt")
        
        rpm_file = open(rpm_file_name, "rb")
        rpm_double_file = open(rpm_double_file_name, "rb")
        his_rpm = pickle.load(rpm_file)
        his_rpm_double = pickle.load(rpm_double_file)
        
        for idx in range(len(his_rpm.buffer)):
            # print(meta)
            (batch_obs, batch_action, batch_reward, batch_next_obs, batch_done, batch_last_action) = his_rpm.buffer[idx]
            cur_rpm.append((batch_obs, batch_action, meta['sim'], batch_next_obs, batch_done, batch_last_action))
            (batch_obs_d, batch_action_d, batch_reward_d, batch_next_obs_d, batch_done_d, batch_last_action_d) = his_rpm_double.buffer[idx]
            cur_rpm_double.append((batch_obs_d, batch_action_d, meta['sim'], batch_next_obs_d, batch_done_d, batch_last_action_d))
            # print(meta['sim'])
            
        
        
        rpm_file.close()
        rpm_double_file.close()
    
    print("len(cur_rpm) = ", len(cur_rpm))
    print("len(cur_rpm_double) = ", len(cur_rpm_double))
        
    cur_rpm_file_name = os.path.join(RPMS_PATH, "his_rpm.txt")
    cur_rpm_double_file_name = os.path.join(RPMS_PATH, "his_rpm_double.txt")
    
    cur_rpm_file = open(cur_rpm_file_name, "wb")
    cur_rpm_double_file = open(cur_rpm_double_file_name, "wb")
    
    # 保存回放内存,写盘很费时间，注意控制写盘频率
    pickle.dump(cur_rpm, cur_rpm_file)
    pickle.dump(cur_rpm_double, cur_rpm_double_file)
    
    cur_rpm_file.close()
    cur_rpm_double_file.close()
        
# make_model()












