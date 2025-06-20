# utils.py
# 鎿嶄綔鏁版嵁搴撶殑鏂囦欢
import csv
import logging
import os
import random
import re
import datetime

import numpy as np
import paramiko as paramiko
import pymysql
import time
from maEnv  import globalValue
import socket
from parl.utils import action_mapping
import sys

################################################
############   涓嶅彲浣跨敤mysql瀹㈡埛绔ￄ1�7   ##############
############ 涓巄uf0tune.cc缁撳悎浣跨敤  ##############
################################################

# 杩滅▼杩炴帴linux, 骞跺疄鐜拌繙绋嬪惎鍔ㄦ暟鎹簄1�7
def sshExe(sys_ip,username,password,cmd):
    client = None
    result = None
    # print(cmd)
    try:
        #鍒涘缓ssh瀹㈡埛绔ￄ1�7
        client = paramiko.SSHClient()
        #绗竴娆sh杩滅▼鏃朵細鎻愮ず杈撳叆yes鎴栬€卬o
        # if globalValue.SSH_CNT <= 5:
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        #瀵嗙爜鏂瑰紡杩滅▼杩炴帴
        client.connect(sys_ip, 22, username=username, password=password, timeout=20)
        #浜掍俊鏂瑰紡杩滅▼杩炴帴
        #key_file = paramiko.RSAKey.from_private_key_file("/root/.ssh/id_rsa")
        #ssh.connect(sys_ip, 22, username=username, pkey=key_file, timeout=20)
        #鎵ц鍛戒抄1�7
        # stdin, stdout, stderr = client.exec_command(cmds)
        stdin, stdout, stderr = client.exec_command(cmd)
        #鑾峰彇鍛戒护鎵ц缁撴灄1�7,杩斿洖鐨勬暟鎹槸涓€涓猯ist
        result = stdout.readlines()
    except Exception as e:
        print(e)
    finally:
        client.close()
        return result


# 灏嗙粰��氬瓧绗︿覆鍙戦€佺粰鏁版嵁搴�
def send_msg_to_server(msg, ip, port):
    p = None
    received_msg = None
    # print('send message-> : ', msg)
    try:
        # 寤虹珛杩炴帴
        # print('msg = {} ip = {} port = {}'.format(msg, ip, port))
        p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        p.connect((ip, port))
        # 鍙戦€佹秷鎭ￄ1�7
        p.send(msg.encode('utf-8'))
        # 鎺ユ敹鍙嶉
        received_msg = p.recv(1024).decode('utf-8')
        # print('received_msg = ',received_msg)
        p.send("exit".encode('utf-8'))
    except Exception as e:
        print(e)
        # print('Meet set variables failure!!')
        p.close()
        return received_msg, False
    # 鍏抽棴杩炴帴
    p.close()
    return received_msg, True

# 璁剧疆缁欏畾鍙傛暟鍊ￄ1�7,鍗冲皢缁欏畾鍊煎皝瑁ￄ1�7
# 鍙傛暟鎸夌収濡備笅椤哄簭鎺掑垄1�7
# [buffer_pool_size, old_blocks_pct, old_threshold_ms, flush_neighbors
#  sleep_time_flush, flush_n,        se_lru_scan_depth, sleep_time_remove,
#  lru_scan_depth,   RESERVE_FREE_PAGE_PCT_FOR_SE, FREE_PAGE_THRESHOLD]
# 灏佽鍚庣殑鏁版嵁鏍煎紡涓猴細set$1.0$2.0$4$3$10000$
def get_set_variables_string(var_names, new_values, node_name, type):
    length = len(var_names)
    s = str(type) + '$'
    if node_name == 'se':
        s = s + 'se'
    else:
        s = s + 'ce'
    # 灏佽鍙傛暟涓暄1�7
    s = s + '$' + str(length)
    # 澧炲姞鍙傛暟濮撳悕鐨勫皝瑁�
    for i in range(length):
        s = s + '$' + var_names[i] + '$' + str(new_values[i])
    print(s)
    return s

# 鎸夌収鍓嶇紑鍒掑垎SE鍜孋E鑺傜偣鍙戦€佺殑鍙傛暄1�7
def get_set_variables_string_nodes(var_names, new_values, type):
    length = len(var_names)
    s_se = str(type) + '$se'
    s_ce = str(type) + '$ce'
    len_se = 0
    len_ce = 0

    # 鍏堢粺璁ￄ1�7'se_'涓暄1�7
    for i in range(length):
        if var_names[i][0] == 's':
            len_se += 1

    len_ce = length - len_se
    # 灏佽鍙傛暟涓暄1�7
    s_se = s_se + '$' + str(len_se)
    s_ce = s_ce + '$' + str(len_ce)

    # 灏佽鍙傛暟鍚嶅拰��瑰簲璁剧疆鍊�
    for j in range(length):
        if j < len_se:
            s_se = s_se + '$' + var_names[j][3:] + '$' + str(new_values[j])
        else:
            s_ce = s_ce + '$' + var_names[j][3:] + '$' + str(new_values[j])
    print(s_se)
    print(s_ce)
    return s_se, s_ce

# 灏哸ction鏢�惧ぇ涓哄彲搴旂敤鐨勫€�
def action_change(env, action):
    action_len = len(action)
    real_action = [0] * action_len
    for k in range(action_len):
        # real_action[k] = int(env.max_info[k] * action[k])
        real_action[k] = action_mapping(action[k], env.min_info[k], env.max_info[k])

    # 淇濊瘉buf_pool_size鎸夌収chunk_size鐨勬暣鏁板€嶈繘琛屾洿鏂ￄ1�7
    # real_action[0] = real_action[0] // globalValue.CHUNK_SIZE * globalValue.CHUNK_SIZE

    real_action = list(map(int, real_action))
    #real_action[0] = int(real_action[0])
    return real_action

def get_default_actions(env):
    default_actions = []
    actions = []
    for info in env.all_actions.keys():
        scales = env.all_actions[info]
        default_actions.append(scales[0])
        actions.append(real_action_to_action(scales[0], scales[1], scales[2]))
    print('default_actions = ', default_actions)
    # env.all_last_action = actions
    if env.ld_adaptor is not None:
        # env.all_last_action =  env.ld_adaptor.reverse_transform(actions)
        # print("[MERGE DEBUG] init action = ", actions)
        # print("[MERGE DEBUG] init action by transform = ", env.ld_adaptor.transform(env.all_last_action))
        # env.ld_adaptor.test(actions)
        env.all_last_action = actions
    else:
        env.all_last_action = actions
    return default_actions

# 渚濊妭鐐逛俊鎭垝鍒嗗苟璁＄畻姣忎釜鑺傜偣鐨勫疄闄呭姩浣渞eal_action[][]
def split_action(env, action):
    real_actions = []
    action_ = []
    last_uuid = 0
    cnt = 0
    # all_actions鏍煎約1�7: 'uuid#innodb_buffer_pool:[default, min, max]'
    for info in env.all_actions.keys():
        # print(info)
        uuid = info.split("#")[0]
        scales = env.all_actions[info]
        # print(scales)
        real_action = action_mapping(action[cnt], scales[1], scales[2])
        if last_uuid != uuid and action_ != []:
            action_ = list(map(int, action_))
            real_actions.append(action_)
            action_ = []
        action_.append(real_action)
        cnt += 1
        last_uuid = uuid
    action_ = list(map(int, action_))
    real_actions.append(action_)
    # real_actions = list(map(int, real_actions))
    return real_actions


def split_default_actions(env):
    real_actions = []
    action = get_default_actions(env)
    action_ = []
    last_uuid = 0
    cnt = 0
    # all_actions鏍煎約1�7: 'uuid#innodb_buffer_pool:[default, min, max]'
    for info in env.all_actions.keys():
        uuid = info.split("#")[0]
        # print('uuid = ', uuid)
        real_action = action[cnt]
        if last_uuid != uuid and action_ != []:
            real_actions.append(action_)
            action_ = []
        action_.append(real_action)
        cnt += 1
        last_uuid = uuid
    real_actions.append(action_)
    # print('real_actions = ', real_actions)
    return real_actions

# 灏唕eal action杞负action鍊�
def real_action_to_action(real_action,low,high):
    action = -1 + (real_action - low) * (2.0 / (high - low))
    action = np.clip(action, -1, 1)
    return action

# def action_to_real_action(action, low, high):
#     real_action = 

# 灏嗕笓��舵帹鑽愯秼鍔胯В鏋愬苟涓巃gent_predict鍚堟垄1�7
def action_with_knowledge(action, action_trend, p, last_action):
    action_len = len(action)
    for k in range(action_len):
        trend = action_trend[k]
        # 鏈夋鐜噋閫夋嫨閬靛畧trend
        rand = np.random.uniform(0, 1)
        # print('rand = ', rand)
        # 闅忔溢�鐢熸垚鐨勬暟钢�藉湪[0,p]鍖洪棿鍐呭嵆涓洪伒��坱rend
        if rand <= p:
            if trend == 1:
                action[k] = np.random.uniform(last_action[k], 1)
            elif trend == -1:
                action[k] = np.random.uniform(-1, last_action[k])

    return action

# 灏哹est_now銆佷笓��舵帹鑽愯秼鍔胯В鏋愬苟涓巃gent_predict鍚堟垄1�7
def action_with_knowledge_and_best_now(action, best_action_now, action_trend, p_best, p_exp, last_action):
    action_len = len(action)
    hit_cnt = 0
    # 姒傜巼p_best閫夋嫨best_action_now, 姒傜巼p_exp閫夋嫨閬靛畧trend,
    rand = np.random.uniform(0, 1)
    for k in range(action_len):
        if action_trend == []:
            trend = 0
        else:
            trend = action_trend[k]
        # 闅忔溢�鐢熸垚鐨勬暟钢�藉湪[0,p]鍖洪棿鍐呭嵆涓洪伒��坱rend
        if trend == 1 and action[k] > last_action[k]:
            hit_cnt += 1
        elif trend == -1 and action[k] < last_action[k]:
            hit_cnt += 1
        elif trend != 0:
            hit_cnt -= 1

        if globalValue.USE_SHAP_COLLECT_DATA:
            rand = np.random.uniform(0, 1)
        
        if 0 < rand <= p_exp:
            if trend == 1 and action[k] < last_action[k]:
                action[k] = np.random.uniform(last_action[k], 1)
            if trend == -1 and action[k] > last_action[k]:
                action[k] = np.random.uniform(-1, last_action[k])
        elif rand <= p_best + p_exp:
            action[k] = best_action_now[k]
    # print("[action_with_knowledge_and_best_now]======================================")
    # print("rand = ", rand)
    # print("action_trend = ", action_trend)
    # print("last_action = ", last_action)
    # print("action = ", action)
    # print("[action_with_knowledge_and_best_now]======================================")
    return action, hit_cnt, rand


# def action_corrext_for_bps(env, action, p_rand, p_exp, action_trend):
#     delta = globalValue.DELTA_ACTION_FOR_ADD
#     if p_rand > p_exp:  
#         return action
    
#     if env.last_action != -2:
#         # 渚濇淇敼node
#         index = 0
#         for se in env.se_info:
#             trend = action_trend[index]
#             # buf_key = str(se.uuid) + '#buffer_pool_size'
#             bp_size_se = action_mapping(action[index], se.tune_action['buffer_pool_size'][1],
#                                         se.tune_action['buffer_pool_size'][2])
#             last_se_bp_size = se.last_bp_size
#             last_action_nor = real_action_to_action(last_se_bp_size, se.tune_action['buffer_pool_size'][1], se.tune_action['buffer_pool_size'][2])
#             if trend == 1 and bp_size_se < last_se_bp_size:
#                 action[index] = np.random.uniform(last_action_nor, min(last_action_nor + delta*abs(last_action_nor), 1))
#             if trend == -1 and bp_size_se > last_se_bp_size:
#                 action[index] = np.random.uniform(max(-1, last_action_nor - delta*abs(last_action_nor)), last_action_nor)
#             index += len(se.tune_action)

#         for ce in env.ce_info:
#             trend = action_trend[index]
#             if ce.is_primary:
#                 if globalValue.LOAD_TYPE == "wo":
#                     rand_wo = np.random.uniform(0, 1)
#                     if rand_wo <= globalValue.RULE_PP:
#                         trend = 1
#             # buf_key = ce.uuid + '#buffer_pool_size'
#             bp_size_ce = action_mapping(action[index], ce.tune_action['buffer_pool_size'][1],
#                                         ce.tune_action['buffer_pool_size'][2])
#             last_ce_bp_size = ce.last_bp_size
#             last_action_nor = real_action_to_action(last_ce_bp_size, ce.tune_action['buffer_pool_size'][1], ce.tune_action['buffer_pool_size'][2])
#             if trend == 1 and bp_size_ce < last_ce_bp_size:
#                 action[index] = np.random.uniform(last_action_nor, min(last_action_nor + delta*abs(last_action_nor), 1))
#             if trend == -1 and bp_size_ce > last_ce_bp_size:
#                 action[index] = np.random.uniform(max(-1, last_action_nor - delta*abs(last_action_nor)), last_action_nor)
            
#             index += len(ce.tune_action)
#     return action
  
  
def action_corrext_for_bps(env, action, p_rand, p_exp, action_trend):
    delta = globalValue.DELTA_ACTION_FOR_ADD
    if p_rand > p_exp:
        return action

    if env.last_action != -2:
        index = 0
        for se in env.se_info:
            trend = action_trend[index]
            # buf_key = str(se.uuid) + '#buffer_pool_size'
            bp_size_se = action_mapping(action[index], se.tune_action['buffer_pool_size'][1],
                                        se.tune_action['buffer_pool_size'][2])
            last_se_bp_size = se.last_bp_size
            last_action_nor = real_action_to_action(last_se_bp_size, se.tune_action['buffer_pool_size'][1], se.tune_action['buffer_pool_size'][2])
            if trend == 1 and (bp_size_se <= last_se_bp_size or abs(bp_size_se-last_se_bp_size)<50000):
                action[index] = np.random.uniform(last_action_nor, min(last_action_nor + delta*abs(last_action_nor+1.2), 1))
            if trend == -1 and (bp_size_se >= last_se_bp_size or abs(bp_size_se-last_se_bp_size)<50000):
                action[index] = np.random.uniform(max(-1, last_action_nor - delta*abs(last_action_nor+1.2)), last_action_nor)
            index += len(se.tune_action)

        total_flag = 0
        flag_index = index
        for ce in env.ce_info:
            total_flag += action_trend[flag_index]
            flag_index += len(ce.tune_action)
        total_flag = 0

        for ce in env.ce_info:
            trend = action_trend[index]
            if ce.is_primary:
                if globalValue.LOAD_TYPE == "wo":
                    rand_wo = np.random.uniform(0, 1)
                    if rand_wo <= globalValue.RULE_PP:
                        trend = 1
            # buf_key = ce.uuid + '#buffer_pool_size'
            bp_size_ce = action_mapping(action[index], ce.tune_action['buffer_pool_size'][1],
                                        ce.tune_action['buffer_pool_size'][2])
            last_ce_bp_size = ce.last_bp_size
            last_action_nor = real_action_to_action(last_ce_bp_size, ce.tune_action['buffer_pool_size'][1], ce.tune_action['buffer_pool_size'][2])
            if total_flag >= 0 and trend == 1 and (bp_size_ce < last_ce_bp_size or abs(bp_size_ce-last_ce_bp_size)<50000):
                action[index] = np.random.uniform(last_action_nor, min(last_action_nor + delta*abs(last_action_nor+1.2), 1))
            if total_flag <= 0 and trend == -1 and (bp_size_ce > last_ce_bp_size or abs(bp_size_ce-last_ce_bp_size)<50000):
                action[index] = np.random.uniform(max(-1, last_action_nor - delta*abs(last_action_nor+1.2)), last_action_nor)

            index += len(ce.tune_action)
    return action

    


def cal_human_hit(action, action_trend, last_action):
    hit_cnt = 0
    hit_list = []
    for k in range(len(action)):
        if action_trend == []:
            trend = 0
        else:
            trend = action_trend[k]
        if trend == 1 and action[k] > last_action[k]:
            hit_cnt += 1
            hit_list.append(1)
        elif trend == -1 and action[k] < last_action[k]:
            hit_cnt += 1
            hit_list.append(1)
        elif trend != 0:
            hit_cnt -= 1
            hit_list.append(-1)
        else:
            hit_list.append(0)
        if k == 0 or k == 17:
                print("[MERGE DEBUG] bps_cur = ", action[k], ", bps_last = ", last_action[k], ", trend = ", trend)
    print("[MERGE DEBUG] hit_list = ", hit_list)
    return hit_cnt

# if __name__ == '__main__':
#     action = -0.98
#     real_action = action_mapping(action, 33554432, 3355443200)
#     print("action=", action)
#     print("real_action=", real_action)
#     action = real_action_to_action(real_action, 33554432, 3355443200)
#     print("real_actionto_action=", action)

# 鍒ゆ柇action鏄惁绗﹢�悎搴旂敤鏉��欢
# 涓嶇鍚堬紝杩斿洖False
# 绗﹀悎锛岃繑鍥濼rue
def action_cirtic(env,action):
    for k in range(len(action)):
        if env.min_info[k] > action[k]:
            return False
    return True


################################################
#############  鍙娇鐢╩ysql瀹㈡埛绔ￄ1�7   ###############
################################################
def create_conn(ip) -> object:

    #杩炴帴閰嶇疆淇℃伄1�7
    # conn_host='192.168.1.102'
    # conn_host='127.0.0.1'
    conn_host=ip
    conn_port=3306
    conn_user='root'
    conn_password='mysql'

    #寤虹珛杩炴帴
    conn = pymysql.connect(host=conn_host, port=conn_port, user=conn_user, password=conn_password)
    return conn


# 鍒涚珛杩炴帴鏁版嵁搴撶殑connection
# 涓巊et_curr涓€璧蜂娇鐢ￄ1�7
# def create_conn() -> object:
#
#     #杩炴帴閰嶇疆淇℃伄1�7
#     # conn_host='192.168.1.102'
#     # conn_host='127.0.0.1'
#     conn_host=globalValue.CONNECT_CE_IP
#     conn_port=3306
#     conn_user='dawn'
#     conn_password='mysql'
#
#     #寤虹珛杩炴帴
#     conn = pymysql.connect(host=conn_host, port=conn_port, user=conn_user, password=conn_password)
#     return conn

# GRANT ALL PRIVILEGES ON *.* TO 'dawn'@'%' identified by 'mysql';
# GRANT SELECT ON performance_schema.* TO 'dawn'@'%';
# GRANT ALL PRIVILEGES ON *.* TO 'prom'@'localhost' identified by 'mysql';
# GRANT SELECT ON performance_schema.* TO 'prom'@'localhost';

# 鑾峰彇鏁版嵁搴撴渶鏂皊tate
def get_current_status():
    if(len(globalValue.GLOBAL_CURRENT_STATUS) == 0):
        return None
    return globalValue.GLOBAL_CURRENT_STATUS

# 璁剧疆鏁版嵁搴撴渶鏂扮姸鎬�
def set_curent_status(current_status):
    globalValue.GLOBAL_CURRENT_STATUS = current_status

# 寤虹珛鏁版嵁搴撹繛鎺ￄ1�7
def get_cur(conn):
    cur = conn.cursor()
    #print('Conntion created successfully!!!')
    return cur

#鍏抽棴鏁版嵁搴撹繛鎺ￄ1�7
def close_conn_mysql(cur, conn):
    cur.close()
    conn.close()
    #print('Conntion closed successfully!!!')


# 鎵цset璇彞锛岃缃弬鏁板€ￄ1�7
# 闇€寤虹珛鏁版嵁搴撹繛鎺ￄ1�7
# 寤鸿杈撳叆涓烘暟缁ￄ1�7
# variables_status = [variable_name, variable_current_value, viriable_max_value, variable_min_value, variable_change_step]
# 鐩墠鏄瓧鍏歌緭鍏ￄ1�7
def execute_sql(cur,knobs_dict):
    #print(knobs_dict)
    for variable_name, variable_value in knobs_dict.items():
        print('variable_name:',variable_name)
        print('variable_value:',variable_value)
        sql = (f'''
                set global {variable_name}={variable_value}
                ''')
        try:
            cur.execute(sql)
        except:
            print('error')
            time.sleep(1)


# 鎵цshow璇彞锛宻how variables like...
# 杈撳叆涓哄崟涓彉閲ￄ1�7
def show_sql(variable_name):
    conn = create_conn()
    cur = get_cur(conn)
    # 寰呮墽琛岃鍙�
    sql = (f'''
            show variables like '{variable_name}'
            ''')
    cur.execute(sql)
    result = cur.fetchall()
    print('The current bp is:', result)
    close_conn_mysql(cur, conn)
    return result


# 鑾峰彇鍙橢�噺褰撳墠鍊硷紝show variables like ...
# 杈撳叆涓烘暟缁勶紝鍙竴娆℃€ц幏鍙栧涓彉閲�
# return result涓烘暟缁ￄ1�7
# [('innodb_buffer_pool_size', '134217728'),
# ('innodb_old_blocks_pct', '37'),
# ('innodb_old_blocks_time', '1000'),
# ('innodb_max_dirty_pages_pct_lwm', '0.000000'),
# ('innodb_flush_neighbors', '1'),
# ('innodb_lru_scan_depth', '1024')]
def show_variables(cur,variable_names):
    result = []
    for variable_name in variable_names:
        temp_result = show_sql(cur,variable_name)
        result.append(temp_result[0])
    return result


# 鑾峰彇鐘舵€佸彉閲忓綋鍓嶅€ￄ1�7,show global status like...
# 杈撳叆涓哄崟涓姸鎬佸彉閲�
# [questions]鐢ㄦ潵璁＄畻reward
def show_status(cur,status_name):
    # 寰呮墽琛岃鍙�
    sql = (f'''
            show global status like '{status_name}'
            ''')
    cur.execute(sql)
    # print("check executed?", cur._check_executed())
    result = cur.fetchall()
    # print("execute '" + sql + "' result = ")
    idx = 0
    for r in result:
        idx += 1
        # print(str(idx) + ": " + str(r))
    sys.stdout.flush()
    return result


# [questions]鐢ㄦ潵璁＄畻reward
def prepare_for_tpcc(ip):
    conn = create_conn(ip)
    cur = get_cur(conn)
    # # 寰呮墽琛岃鍙�
    sql1 = (f'''
            CREATE DATABASE tpcc;
            ''')
    # sql2 = (f'''
    #         source /home/wjx/tpcc-mysql/create_table.sql;
    #             ''')
    # # sql3 = 'create_table.sql;'
    # # sql4 = 'sorce /home/wjx/tpcc-mysql/add_fkey_idx.sql;
    cur.execute(sql1)
    # cur.execute(sql2)
    # cur.execute(sql3)
    # cur.execute(sql4)

    close_conn_mysql(cur, conn)
    return

# 鑾峰彇缂撳啿姹犲ぇ灏ￄ1�7
def get_bps(ip, port):
    bps, flag = send_msg_to_server('4', ip, port)
    print("[{0}:{1}] bps = {2}".format(ip, port, bps))
    bps = int(bps.split("$")[0])
    return bps


# 璁＄畻hit_ratio
def get_se_hr(node):
    ip = node.ip
    port = node.port
    # 鑾峰彇褰撳墠椤甸潰鎬绘暟
    pages_info_before, flag = send_msg_to_server('2', ip, port)
    # 璁＄畄1�75绉掔殑hit_ratio
    time.sleep(5)
    # 鑾峰彇褰撳墠椤甸潰鎬绘暟
    pages_info_after, flag = send_msg_to_server('2',ip,port)

    hr = cal_hr(pages_info_before, pages_info_after)

    return hr

# 浣跨敤sql璇彞璁＄畻hit_ratio
def get_hr():
    conn = create_conn()
    cur = get_cur(conn)
    p1 = show_status(cur, 'innodb_buffer_pool_reads')
    # time.sleep(10)
    p2 = show_status(cur, 'innodb_buffer_pool_read_requests')
    p = 1 - (int(p1[0][1]))/(int(p2[0][1]))
    # print('The current hit_ratio is:',p)
    close_conn_mysql(cur, conn)
    return p

# 浣跨敤sql璇彞璁＄畻hit_ratio
def get_node_hr(node):
    conn = create_conn(node.ip)
    cur = get_cur(conn)
    p1 = show_status(cur, 'innodb_buffer_pool_reads')
    # time.sleep(10)
    p2 = show_status(cur, 'innodb_buffer_pool_read_requests')
    p = 1 - (int(p1[0][1]))/(int(p2[0][1]))
    # print('The current hit_ratio is:',p)
    close_conn_mysql(cur, conn)
    return p

def cal_hr(pages_info_before, pages_info_after):
    pages1 = pages_info_before.split("$")
    instances = int(pages1[0])
    len = 2*instances+1

    pages1 = list(map(int, pages1[0:len]))
    pages2 = list(map(int, pages_info_after.split("$")[0:len]))

    total_pages = 0.0
    read_pages = 0.0

    for i in range(instances):
        # print("i=",i)
        total_pages = pages2[i+1] - pages1[i+1] + total_pages
        # print("i+1+instances = ", i+1+instances)
        read_pages = pages2[i+1+instances] - pages1[i+1+instances] + read_pages

    if total_pages == 0:
        hit_ratio = -1
    else:
        hit_ratio = (total_pages - read_pages) / total_pages

    return hit_ratio




# 璁＄畻qps --- calculate average qps
# 10s
def get_qps():
    conn = create_conn()
    cur = get_cur(conn)
    p1 = show_status(cur, 'questions')
    time.sleep(10)
    p2 = show_status(cur, 'questions')
    time.sleep(10)
    p3 = show_status(cur, 'questions')
    p_1 = (int(p2[0][1]) - int(p1[0][1])) / 10.0
    p_2 = (int(p3[0][1]) - int(p2[0][1])) / 10.0
    p = (p_1 + p_2) / 2.0
    # print('The current qps is:',p)
    close_conn_mysql(cur, conn)
    return p

# 璁＄畻qps--- calculate average qps
# 10s
def get_node_qps(node):
    t = 30.0
    conn = create_conn(node.ip)
    cur = get_cur(conn)
    p1 = show_status(cur, 'questions')
    time.sleep(t)
    p2 = show_status(cur, 'questions')
    time.sleep(t)
    p3 = show_status(cur, 'questions')
    p_1 = (int(p2[0][1]) - int(p1[0][1])) / t
    p_2 = (int(p3[0][1]) - int(p2[0][1])) / t
    p = (p_1 + p_2) / 2.0
    # print('The current qps is:',p)
    close_conn_mysql(cur, conn)
    return p


def cal_r(delta_t0, delta_t1):
    if delta_t0 > 0:
        r = abs(1 + delta_t1) * (pow(1 + delta_t0, 2) - 1)
    else:
        r = (-1) * abs(1 - delta_t1) * (pow(1 - delta_t0, 2) - 1)
    if r > 0 and delta_t1 < 0:
        r = 0
    return r

def cal_reward_ce_single_node(node, h_before , h_after , q_before , q_after , action_after):
    # 鍏堣绠梔elta(t-t0)鍜宒elta(t-t-1)
    print("[DEBUG][cal_reward_ce_single_node] cal delta_h0")
    delta_h0 = (h_after - node.hit_t0) / node.hit_t0
    print("[DEBUG][cal_reward_ce_single_node] cal delta_q0")
    delta_q0 = (q_after - node.qps_t0) / node.qps_t0
    print("[DEBUG][cal_reward_ce_single_node] cal delta_ht, h_before = {0}".format(h_before))
    delta_ht = (h_after - h_before) / h_before
    print("[DEBUG][cal_reward_ce_single_node] cal delta_qt, q_before = {0}".format(q_before))
    delta_qt = (q_after - q_before) / q_before
    node.delta_h0 = delta_h0
    node.delta_ht = delta_ht
    node.delta_q0 = delta_q0
    node.delta_qt = delta_qt

    # print('h_before ', h_before)
    # print('h_after ', h_after)
    # print('qps_before ', q_before)
    # print('qps_after ', q_after)

    # 璁＄畻qps鍜宧it_ratio瀵瑰簲鐨剄1�7(浠ヨ繃绋嬩负涓诲锛ￄ1�7
    # if delta_h0 > 0:
    #     rewards_h = abs(1 + delta_ht) * (pow(1 + delta_h0, 2) - 1)
    # else:
    #     rewards_h = (-1) * abs(1 - delta_h0) * (pow(1 - delta_ht, 2) - 1)
    print("[DEBUG][cal_reward_ce_single_node] cal rewards_h")
    rewards_h = cal_r(delta_h0, delta_ht)
    # if rewards_h > 0 and delta_h0 < 0:
    #     rewards_h = 0
    # if delta_q0 > 0:
    #     rewards_q = abs(1 + delta_q0) * (pow(1 + delta_qt, 2) - 1)
    # else:
    #     rewards_q = (-1) * abs(1 - delta_q0) * (pow(1 - delta_qt, 2) - 1)
    print("[DEBUG][cal_reward_ce_single_node] cal rewards_q")
    rewards_q = cal_r(delta_q0, delta_qt)
    # if rewards_q > 0 and delta_q0 < 0:
    #     rewards_q = 0

    ## 澧炲姞缂撳啿姹犺祫婧愮殑鑰冭檻锛宎ction鐨勬璐熷苟涓嶈兘澶熺洿鎺ュ弽鏄犵紦鍐叉睜璋冨ぇ杩樻槸璋冨皬
    bp_size_after = action_after
    bp_size_before = node.bpsize_before
    print("[DEBUG][cal_reward_ce_single_node] over")
    delta_b0 = (bp_size_after - node.bp_size_0) / node.bp_size_0
    delta_bt = (bp_size_after - bp_size_before) / bp_size_before
    
    print("[DEBUG][cal_reward_ce_single_node] return")
    # print('bps_before ', bp_size_before)
    # print('bps_after ', bp_size_after)
    # if delta_bt > 0:
    #     rewards_b = abs(1+delta_b0) * (pow(1+delta_bt, 2) - 1)
    # else:
    #     rewards_b = (-1) * abs(1-delta_b0) * (pow(1-delta_bt, 2) - 1)
    # if delta_b0 <= 0 and delta_q0 >= 0 and delta_h0 >= 0:
    #     rewards_b = 3
    # else:
    #     rewards_b = -2

    return rewards_q, rewards_h, delta_q0

    # 璁剧疆鏉冮噸
    # wh = 0.2
    # wq = 0.5
    # wb = 0.3
    # wh = 0.1
    # wq = 0.6
    # wb = 0.3
    wh = 0.15
    wq = (1-wh) / 2.0
    wb = wq

    # 璁＄畻��為檯鐨勫鍔�
    reward = rewards_h * wh + rewards_q * wq + rewards_b * wb
    if reward < -1:
        return -1
    elif reward > 1:
        return 1
    else:
        return reward

def cal_reward_ce_my(env, h_before , h_after , q_before , q_after , action_after, min_info, max_info):
    # 鍏堣绠梔elta(t-t0)鍜宒elta(t-t-1)
    delta_h0 = (h_after - env.hit_t0)/env.hit_t0
    delta_q0 = (q_after - env.qps_t0)/env.qps_t0
    delta_ht = (h_after - h_before)/h_before
    delta_qt = (q_after - q_before)/q_before

    # print('h_before ', h_before)
    # print('h_after ', h_after)
    # print('qps_before ', q_before)
    # print('qps_after ', q_after)

    #璁＄畻qps鍜宧it_ratio瀵瑰簲鐨剄1�7(浠ヨ繃绋嬩负涓诲锛ￄ1�7
    # if delta_ht > 0:
    #     rewards_h = abs(1+delta_h0) * (pow(1+delta_ht, 2) - 1)
    # else:
    #     rewards_h = (-1) * abs(1-delta_h0) * (pow(1-delta_ht, 2) - 1)
    #
    # if delta_qt > 0:
    #     rewards_q = abs(1+delta_q0) * (pow(1+delta_qt, 2) - 1)
    # else:
    #     rewards_q = (-1) * abs(1-delta_q0) * (pow(1-delta_qt, 2) - 1)

    if delta_h0 > 0:
        rewards_h = abs(1+delta_ht) * (pow(1+delta_h0, 2) - 1)
    else:
        rewards_h = (-1) * abs(1-delta_ht) * (pow(1-delta_h0, 2) - 1)

    if delta_q0 > 0:
        rewards_q = abs(1+delta_qt) * (pow(1+delta_q0, 2) - 1)
    else:
        rewards_q = (-1) * abs(1-delta_qt) * (pow(1-delta_q0, 2) - 1)


    # if rewards_q > 0 and delta_q0 < 0:
    #     rewards_q = 0

    ## 澧炲姞缂撳啿姹犺祫婧愮殑鑰冭檻锛宎ction鐨勬璐熷苟涓嶈兘澶熺洿鎺ュ弽鏄犵紦鍐叉睜璋冨ぇ杩樻槸璋冨皬
    bp_size_after = action_mapping(action_after, min_info[0], max_info[0])
    bp_size_before = env.bpsize_before
    delta_b0 = (bp_size_after - env.bp_size_0) / env.bp_size_0
    delta_bt = (bp_size_after - bp_size_before) / bp_size_before
    # print('bps_before ', bp_size_before)
    # print('bps_after ', bp_size_after)
    # if delta_bt > 0:
    #     rewards_b = abs(1+delta_b0) * (pow(1+delta_bt, 2) - 1)
    # else:
    #     rewards_b = (-1) * abs(1-delta_b0) * (pow(1-delta_bt, 2) - 1)
    if delta_bt <= 0 and delta_b0 <= 0:
        rewards_b = 1
    else:
        rewards_b = -1

    #璁剧疆鏉冮噸
    # wh = 0.2
    # wq = 0.5
    # wb = 0.3
    # wh = 0.1
    # wq = 0.6
    # wb = 0.3
    wh = 0.1
    wq = 0.6
    wb = 0.3

    #璁＄畻��為檯鐨勫鍔�
    reward = rewards_h * wh + rewards_q * wq + rewards_b * wb
    if reward < -1:
        return -1
    elif reward > 1:
        return 1
    else:
        return reward

def cal_reward_se_single_node(node, h_before, h_after, action_after):
    # 鍏堣绠梔elta(t-t0)鍜宒elta(t-t-1)
    # print('node{} calr {} {} {}'.format(node.uuid, h_befor e, h_after, action_after))
    delta_h0 = (h_after - node.hit_t0) / node.hit_t0
    delta_ht = (h_after - h_before) / h_before
    node.delta_h0 = delta_h0
    node.delta_ht = delta_ht

    # print('h_before ', h_before)
    # print('h_after ', h_after)
    # print('qps_before ', q_before)
    # print('qps_after ', q_after)

    # 璁＄畻qps鍜宧it_ratio瀵瑰簲鐨剄1�7(浠ヨ繃绋嬩负涓诲1�7????锛�
    # if delta_ht > 0:
    #     rewards_h = abs(1 + delta_h0) * (pow(1 + delta_ht, 2) - 1)
    # else:
    #     rewards_h = (-1) * abs(1 - delta_h0) * (pow(1 - delta_ht, 2) - 1)

    # if delta_h0 > 0:
    #     rewards_h = abs(1 + delta_ht) * (pow(1 + delta_h0, 2) - 1)
    # else:
    #     rewards_h = (-1) * abs(1 - delta_ht) * (pow(1 - delta_h0, 2) - 1)
    rewards_h = cal_r(delta_h0, delta_ht)

    ## 澧炲姞缂撳啿姹犺祫婧愮殑鑰冭檻锛宎ction鐨勬璐熷苟涓嶈兘澶熺洿鎺ュ弽鏄犵紦鍐叉睜璋冨ぇ杩樻槸璋冨皬
    bp_size_after = action_after
    bp_size_before = node.bpsize_before
    delta_b0 = (bp_size_after - node.bp_size_0) / node.bp_size_0
    delta_bt = (bp_size_after - bp_size_before) / bp_size_before
    # print('bps_before ', bp_size_before)
    # print('bps_after ', bp_size_after)
    # if delta_bt > 0:
    #     rewards_b = abs(1+delta_b0) * (pow(1+delta_bt, 2) - 1)
    # else:
    #     rewards_b = (-1) * abs(1-delta_b0) * (pow(1-delta_bt, 2) - 1)
    # if delta_b0 <= 0 and delta_h0 >= 0:
    #     rewards_b = 1
    # else:
    #     rewards_b = -1

    return rewards_h

    # 璁剧疆鏉冮噸
    wh = 0.4
    wb = 0.6

    # 璁＄畻��為檯鐨勫鍔�
    reward = rewards_h * wh + rewards_b * wb
    if reward < -1:
        return -1
    elif reward > 1:
        return 1
    else:
        return reward

def cal_reward_se_my(env, h_before, h_after, action_after, min_info, max_info):
    # 鍏堣绠梔elta(t-t0)鍜宒elta(t-t-1)
    delta_h0 = (h_after - env.hit_t0)/env.hit_t0
    delta_ht = (h_after - h_before)/h_before

    # print('h_before ', h_before)
    # print('h_after ', h_after)
    # print('qps_before ', q_before)
    # print('qps_after ', q_after)

    #璁＄畻qps鍜宧it_ratio瀵瑰簲鐨剄1�7(浠ヨ繃绋嬩负涓诲锛ￄ1�7
    if delta_ht > 0:
        rewards_h = abs(1+delta_h0) * (pow(1+delta_ht, 2) - 1)
    else:
        rewards_h = (-1) * abs(1-delta_h0) * (pow(1-delta_ht, 2) - 1)

    ## 澧炲姞缂撳啿姹犺祫婧愮殑鑰冭檻锛宎ction鐨勬璐熷苟涓嶈兘澶熺洿鎺ュ弽鏄犵紦鍐叉睜璋冨ぇ杩樻槸璋冨皬
    bp_size_after = action_mapping(action_after, min_info[0], max_info[0])
    bp_size_before = env.bpsize_before
    delta_b0 = (bp_size_after - env.bp_size_0) / env.bp_size_0
    delta_bt = (bp_size_after - bp_size_before) / bp_size_before
    # print('bps_before ', bp_size_before)
    # print('bps_after ', bp_size_after)
    # if delta_bt > 0:
    #     rewards_b = abs(1+delta_b0) * (pow(1+delta_bt, 2) - 1)
    # else:
    #     rewards_b = (-1) * abs(1-delta_b0) * (pow(1-delta_bt, 2) - 1)
    if delta_bt <= 0 and delta_b0 <= 0:
        rewards_b = 1
    else:
        rewards_b = -1

    #璁剧疆鏉冮噸
    wh = 0.5
    wb = 0.5

    #璁＄畻��為檯鐨勫鍔�
    reward = rewards_h * wh + rewards_b * wb
    if reward < -1:
        return -1
    elif reward > 1:
        return 1
    else:
        return reward

def cal_reward_ce_1(env, h_before , h_after , q_before , q_after):
    # 鍏堣绠梔elta(t-t0)鍜宒elta(t-t-1)
    delta_h0 = (h_after - env.hit_t0)/env.hit_t0
    delta_q0 = (q_after - env.qps_t0)/env.qps_t0
    delta_ht = (h_after - h_before)/h_before
    delta_qt = (q_after - q_before)/q_before

    # print('delta_h0', delta_h0)
    # print('delta_ht', delta_ht)
    # print('delta_q0', delta_q0)
    # print('delta_qt', delta_qt)

    # rate0 = 0.02
    # rate1 = 0.08
    # #璁＄畻qps鍜宧it_ratio瀵瑰簲鐨剄1�7(浠ヨ繃绋嬩负涓诲锛ￄ1�7
    # rewards_h = (rate0 * delta_h0) + (rate1 * delta_ht)
    # rewards_q = (rate0 * delta_q0) + (rate1 * delta_qt)

    # 璁＄畻qps鍜宧it_ratio瀵瑰簲鐨剄1�7(浠ヨ繃绋嬩负涓诲锛ￄ1�7
    # TODO:璁＄畻涓€涓寚鏍囩殑reward鍙互鍗曠嫭鍐欎竴涓嚱鏁癱al_reward_from()
    if delta_ht > 0:
        rewards_h = abs(1 + delta_h0) * (pow(1 + delta_ht, 2) - 1)
    else:
        rewards_h = (-1) * abs(1 - delta_h0) * (pow(1 - delta_ht, 2) - 1)
    if rewards_h > 0 and delta_ht < 0:
        rewards_h = 0

    if delta_qt > 0:
        rewards_q = abs(1 + delta_q0) * (pow(1 + delta_qt, 2) - 1)
    else:
        rewards_q = (-1) * abs(1 - delta_q0) * (pow(1 - delta_qt, 2) - 1)

    if rewards_q > 0 and delta_qt < 0:
        rewars_q = 0

    # print('rewards_h', rewards_h)
    # print('rewards_q', rewards_q)
    # if rewards_q > 0 and delta_q0 < 0:
    #     rewards_q = 0

    ## 澧炲姞缂撳啿姹犺祫婧愮殑鑰冭檄1�7


    #璁剧疆鏉冮噸
    wh = 0
    wq = 1

    #璁＄畻��為檯鐨勫鍔�
    reward = rewards_h * wh + rewards_q * wq
    return reward


def cal_reward_ce(env, h_before, h_after, action_before, action_after):
    # reward = 0
    # 鍏堝鐞嗙紦瀛樺懡涓巼
    # h_before = round(h_before, 3)
    # h_after = round(h_after, 3)
    # delta = round((h_after - h_before), 2)
    # delta_bps = 0
    # if delta < 0:
    #     reward = 0
    # elif delta > 0:
    #     hps_r = delta * 1000
    #     reward = hps_r
    # else:
    #     delta_bps = b_after - b_before
    #
    #     if delta_bps >= 0:
    #         reward = 0
    #     else:
    #         reward = delta_bps / globalValue.MAX_POOL_SIZE
    #         reward = -reward * 100
    # print('delta_hit_ratio: ', delta, '    delta_bps: ', delta_bps, '    reward: ', reward)
    h_after = round(h_after, 2)

    delta_hit_ratio = h_after - env.max_hit_ratio
    #delta_bps = round((action_after - action_before), 2)
    #
    if delta_hit_ratio >= 0:
        env.max_hit_ratio = h_after
        #reward = abs(delta_bps)
    #    reward = 1
    #elif delta_hit_ratio == 0:
        #reward = -delta_bps
        reward = -abs(action_after + 1) / 2
    elif delta_hit_ratio >= -0.011:
        reward = -abs(action_after + 1) / 2
    else:
        #reward = -abs(delta_bps)
        reward = -1

    # h_after = round(h_after, 2)
    # if h_after > env.max_hit_ratio:
    #    env.max_hit_ratio = h_after
    # r = b_after / globalValue.MAX_POOL_SIZE
    # if(h_after == env.max_hit_ratio):
    #    reward = -r
    # else:
    #    reward = r
    #
    # reward = 10 * reward





    # hps_r = (round(h_after, 2) - 0.80) * 10
    # bps_r = (b_after / globalValue.MAX_POOL_SIZE) * 10

    # if h_after > env.max_hit_ratio:
    #     env.max_hit_ratio = h_after
    #     reward =
    # else:




    # reward = 0.8 * hps_r - 0.2 * bps_r
    #
    # if b_after == 31457280:
    #     reward = -1000
    #
    #
    # print('h_after = ', h_after, 'reward = ', reward)




    # if b_after > 838860800:
    #     reward = (1 - b_after /( globalValue.MAX_POOL_SIZE+100)) * 10
    # elif b_after < 209715200:
    #     reward = (b_after / globalValue.MAX_POOL_SIZE - 1) * 10
    # else:
    #     reward = (b_after / globalValue.MAX_POOL_SIZE) * 10



    # hps_r = (round(h_after, 2) - 0.80) * 10
    # # bps_r = (b_after / globalValue.MAX_POOL_SIZE) * 10
    # # reward = 0.8 * hps_r - 0.2 * bps_r
    # reward = hps_r

    #######################################################################################
    # 浠ヤ笅涓哄彧璋冩暣缂撳啿姹犲ぇ灏忓彲浠ヨ幏寰楁帹鑽愬€肩殑濂栧姳鍑芥暄1�7  0414
    # if h_after == -1:
    #     return 0
    #
    # h_before = round(env.max_hit_ratio, 3)
    # h_after = round(h_after, 3)
    # delta = round((h_after - h_before), 2)
    # delta_bps=action_after-action_before
    #
    # if delta < 0:
    #    reward = -1
    # elif delta > 0:
    #    env.max_hit_ratio = h_after
    #    reward = 1
    # else:
    #    reward = 0
    #######################################################################################

    # if delta<0:
    #     reward=-1.0
    # elif delta>0:
    #     reward=1.0
    # else:
    #     if delta_bps>0:
    #         reward=-1.0
    #     elif delta_bps<0:
    #         reward=0.0
    #     else:
    #         reward=0.0

    # delta_bps = delta_bps = b_after - b_before
    # delta_bps = (b_after - b_before) / globalValue.MAX_POOL_SIZE
    # delta_action = action_after - action_before
    #if delta < 0:
    #    reward = -abs(delta) * 500
    #elif delta > 0:
    #    env.max_hit_ratio = h_after
        #     hps_r = delta * 1000
    #    reward = abs(delta) * 500
    #else:
    #    reward = -abs(action_after) * 50

    print('h_after = ', h_after, 'action_after = ', action_after, 'current_max_hit_ratio = ', env.max_hit_ratio, 'reward = ', reward)
    return reward




    # hps_r = delta * 10
    # bps_r = (b_after - b_before) / globalValue.MAX_POOL_SIZE
    #
    # reward = 1000 * (0.8 * hps_r - 0.2 * bps_r)
    #
    # print('delta = ', delta)

    # delta = q_after - q_before
    # if abs(delta) <= 50:
    #     qps_r = 0
    # else:
    #     qps_r = delta / 1000.0
    #
    # bps_r = (b_after - b_before) / globalValue.MAX_POOL_SIZE
    #
    # reward = 0.8 * qps_r - 0.2 * bps_r

    # if value2 > value1:
    #    qps_r = 1
    # elif value2 == value1:
    #    qps_r = 0
    # else:
    #    qps_r = -1


    # qps_r = value2 - value1
    #
    # bps_r = value3 / globalValue.MAX_POOL_SIZE * 50
    #
    # if abs(qps_r) < 50:
    #     qps_r = 0
    #
    # reward = qps_r - bps_r

    # print('hit_ratio reward = ', hps_r, 'bps reward = ', bps_r, 'reward = ', reward)
    #
    # return reward


def cal_reward_se(value1, value2, value3):
    if value2 > value1:
        qps_r = 1
    elif value2 == value1:
        qps_r = 0
    else:
        qps_r = -1

    bps_r = value3 / globalValue.MAX_POOL_SIZE

    reward = qps_r - bps_r

    return reward

def load_bash_remote(type):
    # load 80s
    # print('LOAD BASH++++')
    timestp = time_to_str(get_timestamp())
    file_name = globalValue.TEST_RES_FILE + 'res' + timestp + '.txt'
    # make_file_cmd = 'touch ' + file_name
    to_file = '> %s 2>%s &' % (file_name, file_name)
    if type == 'sysbench':
        cmd = globalValue.LOAD_BASH
        # print(cmd)
        flag = list()
        for i in range(1):
            tmp_flag = sshExe(globalValue.CONNECT_CE_IP[i], globalValue.SSH_USERNAME, globalValue.SSH_PASSWD, cmd)
            flag = flag.append(tmp_flag)
    elif type == 'tpcc':
        cmd = globalValue.LOAD_TPCC
        # prepare_for_tpcc(globalValue.CONNECT_CE_IP[0])
        # print(cmd)
        flag = sshExe(globalValue.CONNECT_CE_IP[0], globalValue.SSH_USERNAME, globalValue.SSH_PASSWD, cmd)
    return flag

def start_for_a_fresh_node(node_start_cmd, new_data_cmd):
    # sleep
    sleep_cmd = 'sleep 1;'
    # newdata
    newdata = new_data_cmd + ';'
    # start with exact string
    start_node = node_start_cmd
    text = sleep_cmd + newdata + start_node
    return text

def start_for_a_not_fresh_node(node_close_cmd, node_start_cmd):
    # sleep
    sleep_cmd = 'sleep 1;'
    # close
    close_node = node_close_cmd + ';'
    # start with exact string
    start_node = node_start_cmd
    text = close_node + sleep_cmd + start_node
    return text

def load_bash(bash_time):
    f = globalValue.TMP_FILE
    if not os.path.exists(f):
        os.mkdir(f)
    f += 'sysbench_result_{}.log'.format(int(time.time()))
    cmd = 'touch ' + f
    os.system(cmd)

    test_f = globalValue.TMP_FILE + 'test.sh'
    cmd = 'touch ' + test_f
    os.system(cmd)
    buffer_time = 30
    bash_time = bash_time + buffer_time

    s = '# !/bin/bash\n'\
        'nohup '\
        '/usr/local/bin/sysbench /usr/local/share/sysbench/oltp_read_only.lua ' \
        '--mysql-user=dawn --mysql-password=mysql ' \
        '--mysql-host=%s ' \
        '--mysql-port=3306 ' \
        '--mysql-db=test --mysql-storage-engine=innodb ' \
        '--table-size=10000 ' \
        '--tables=100 ' \
        '--threads=32 ' \
        '--events=0 ' \
        '--report-interval=10 ' \
        '--range_selects=off ' \
        '--time=%d run ' \
        '> %s ' \
        '2>%s &' \
        % (globalValue.CONNECT_MA_IP, bash_time, f, f)
    with open(test_f, 'w') as fs:
        fs.write(s)

    # print('bash_time ', bash_time)

    # 璁板綍璁粌寮€濮嬫椂闂ￄ1�7
    # globalValue.EPISODE_START_TIME = time.time()
    os.system('sh ' + test_f)
    print('bash_end ', f)
    return f, buffer_time + 10



def parse_tpcc(file_path):
    with open(file_path) as f:
        lines = f.read()
    temporal_pattern = re.compile(".*?trx: (\d+.\d+), 95%: (\d+.\d+), 99%: (\d+.\d+), max_rt:.*?")
    temporal = temporal_pattern.findall(lines)
    tps = 0
    latency = 0
    qps = 0

    for i in temporal[-10:]:
        tps += float(i[0])
        latency += float(i[2])
    num_samples = len(temporal[-10:])
    tps /= num_samples
    latency /= num_samples
    # interval
    tps /= 1
    return [tps, latency, qps]

def parse_sysbench(file_path):
    with open(file_path) as f:
        lines = f.read()
    temporal_pattern = re.compile(
        "tps: (\d+.\d+) qps: (\d+.\d+) \(r/w/o: (\d+.\d+)/(\d+.\d+)/(\d+.\d+)\)" 
        " lat \(ms,95%\): (\d+.\d+) err/s: (\d+.\d+) reconn/s: (\d+.\d+)")
    temporal = temporal_pattern.findall(lines)
    tps = 0
    latency = 0
    qps = 0

    for i in temporal[-10:]:
        tps += float(i[0])
        latency += float(i[5])
        qps += float(i[1])
    num_samples = len(temporal[-10:])
    tps /= num_samples
    qps /= num_samples
    latency /= num_samples
    return [tps, latency, qps]

def record_best(qps, hit_ratio, bps):
    filename = 'bestnow.log'
    best_flag = False
    if os.path.exists(globalValue.BEST_FILE + filename):
        qps_best = qps
        hit_ratio_best = hit_ratio
        bps_best = bps
        if hit_ratio_best != 0 and qps_best != 0:
            # with open(globalValue.BEST_FILE + filename) as f:
            #     lines = f.readlines()
            # best_now = lines[0].split(',')
            # if qps_best >= float(best_now[0]) and hit_ratio_best >= float(best_now[1]) and bps_best <= float(best_now[2]):
            if globalValue.REWARD_NOW >= globalValue.MAX_REWARD:
                best_flag = True
                with open(globalValue.BEST_FILE + filename, 'w') as f:
                    f.write(str(qps_best) + ',' + str(hit_ratio_best) + ',' + str(bps_best))
    else:
        with open(globalValue.BEST_FILE + filename, 'w') as file:
            qps_best = qps
            hit_ratio_best = hit_ratio
            bps_best = bps
            file.write(str(qps_best) + ',' + str(hit_ratio_best) + ',' + str(bps_best))
            best_flag = True
    return best_flag

def record_best_nodes(q_after_ce, h_after_ce, bps_ce, h_after_se, bps_se):
    filename = 'bestnow.log'
    best_flag = False
    if os.path.exists(globalValue.BEST_FILE + filename):
        if q_after_ce != 0:
            if globalValue.REWARD_NOW >= globalValue.MAX_REWARD:
                best_flag = True
                with open(globalValue.BEST_FILE + filename, 'w') as f:
                    f.write(str(q_after_ce) + ',' + str(h_after_ce) + ',' + str(bps_ce) + ',' + str(h_after_se) + ',' + str(bps_se))
    else:
        with open(globalValue.BEST_FILE + filename, 'w') as file:
            file.write(str(q_after_ce) + ',' + str(h_after_ce) + ',' + str(bps_ce) + ',' + str(h_after_se) + ',' + str(bps_se))
            best_flag = True
    return best_flag

def record_all_best(ses, ces):
    filename = 'bestnow.log'
    best_flag = False
    
    # all ce qps
    q_after_total = 0
    for ce in ces:
        q_after_total += ce.last_qps
    
    if os.path.exists(globalValue.BEST_FILE + filename):
        if ces[0].last_qps != 0 and globalValue.REWARD_NOW >= globalValue.MAX_REWARD:
            best_flag = True
            with open(globalValue.BEST_FILE + filename, 'w') as f:
                for ce in ces:
                    if ce.is_primary == True:
                        # all ce qps
                        # f.write(
                        #     str(ce.last_qps) + ',' + str(ce.last_hr) + ',' + str(ce.last_bp_size))
                        f.write(
                            str(q_after_total) + ',' + str(ce.last_hr) + ',' + str(ce.last_bp_size))
                    else:
                        f.write(
                            ',' + str(ce.last_hr) + ',' + str(ce.last_bp_size))
                for se in ses:
                    f.write(
                        ',' + str(se.last_hr) + ',' + str(se.last_bp_size))
    else:
        with open(globalValue.BEST_FILE + filename, 'w') as f:
            best_flag = True
            for ce in ces:
                if ce.is_primary == True:
                    # all ce qps
                    # f.write(
                    #     str(ce.last_qps) + ',' + str(ce.last_hr) + ',' + str(ce.last_bp_size))
                    f.write(
                        str(q_after_total) + ',' + str(ce.last_hr) + ',' + str(ce.last_bp_size))
                else:
                    f.write(
                        ',' + str(ce.last_hr) + ',' + str(ce.last_bp_size))
            for se in ses:
                f.write(
                    ',' + str(se.last_hr) + ',' + str(se.last_bp_size))
    record_all_best_all_record(ses, ces)
    return best_flag


def record_all_best_all_record(ses, ces):
    filename = 'bestall.log'
    best_flag = False
    
    # all ce qps
    q_after_total = 0
    for ce in ces:
        q_after_total += ce.last_qps
        
        
    if os.path.exists(globalValue.BEST_FILE + filename):
        if ces[0].last_qps != 0 and globalValue.REWARD_NOW >= globalValue.MAX_REWARD:
            best_flag = True
            with open(globalValue.BEST_FILE + filename, 'a') as f:
                for ce in ces:
                    if ce.is_primary == True:
                        # all ce qps
                        # f.write(
                        #     str(ce.last_qps) + ',' + str(ce.last_hr) + ',' + str(ce.last_bp_size))
                        f.write(
                            str(q_after_total) + ',' + str(ce.last_hr) + ',' + str(ce.last_bp_size))
                    else:
                        f.write(
                            ',' + str(ce.last_hr) + ',' + str(ce.last_bp_size))
                for se in ses:
                    f.write(
                        ',' + str(se.last_hr) + ',' + str(se.last_bp_size))
                f.write('\n')
    else:
        with open(globalValue.BEST_FILE + filename, 'a') as f:
            best_flag = True
            for ce in ces:
                if ce.is_primary == True:
                    # all ce qps
                        # f.write(
                        #     str(ce.last_qps) + ',' + str(ce.last_hr) + ',' + str(ce.last_bp_size))
                        f.write(
                            str(q_after_total) + ',' + str(ce.last_hr) + ',' + str(ce.last_bp_size))
                else:
                    f.write(
                        ',' + str(ce.last_hr) + ',' + str(ce.last_bp_size))
            for se in ses:
                f.write(
                    ',' + str(se.last_hr) + ',' + str(se.last_bp_size))
            f.write('\n')
    return best_flag



def get_best_now(filename):
    with open(globalValue.BEST_FILE + filename) as f:
        lines = f.readlines()
    best_now = lines[0].split(',')
    return [float(best_now[0]), float(best_now[1]), float(best_now[2])]

def get_best_now_nodes(filename):
    with open(globalValue.BEST_FILE + filename) as f:
        lines = f.readlines()
    best_now = lines[0].split(',')
    return [float(best_now[0]), float(best_now[1]), float(best_now[2]), float(best_now[3]), float(best_now[4])]

def get_best_now_all_nodes(filename):
    with open(globalValue.BEST_FILE + filename) as f:
        lines = f.readlines()
    best_now = lines[0].split(',')
    return best_now

def handle_csv(src, dest, num, n_labels=2):
    with open(src, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        rewards = [abs(float(row[num])) for row in rows[1:] if row[num]]  # 璺宠繃鏍囬琛屽苟鏀堕泦濂栧姄1�7
        
        # 璁＄畻鍒嗙晫鐐�
        rewards = sorted(rewards)
        min_val, max_val = rewards[0], rewards[-1]
        thresholds = [min_val + i*(max_val - min_val)/n_labels for i in range(1, n_labels)]
        
        # 鏇存柊濂栧姳鍊�
        for i, row in enumerate(rows[1:], start=1):
            reward = abs(float(row[num]))
            # 纭畾褰撳墠濂栧姳灞炰簬鍝釜鍖洪棿
            label = next((idx for idx, threshold in enumerate(thresholds) if reward < threshold), n_labels - 1)
            rows[i][num] = label
    
    # 鍐欏叆鏂版枃浠�
    with open(dest, 'w', newline='', encoding='utf-8') as file_obj:
        writer = csv.writer(file_obj)
        writer.writerows(rows)



def get_timestamp():
    """
    鑾峰彇UNIX鏃堕棿鎴ￄ1�7
    """
    return int(time.time())

def time_to_str(timestamp):
    """
    灏嗘椂闂存埑杞崲鎴怺YYYY-MM-DD HH:mm:ss]鏍煎約1�7
    """
    return datetime.datetime.\
        fromtimestamp(timestamp).strftime("%Y-%m-%d_%H:%M:%S")

# Logger utils
class Logger:

    def __init__(self, name, log_file=''):
        self.log_file = log_file
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        sh = logging.StreamHandler()
        self.logger.addHandler(sh)
        self.start_time = get_timestamp()
        self.end_time = get_timestamp()
        print('LOG_FILE :', log_file)
        with open(self.log_file, 'w+') as f:
            f.write('=====log============' + '\n')

        if len(log_file) > 0:
            self.log2file = True
        else:
            self.log2file = False

    def _write_file(self, msg):
        if self.log2file:
            with open(self.log_file, 'a+') as f:
                f.write(msg + '\n')

    def get_timestr(self):
        timestamp = get_timestamp()
        date_str = time_to_str(timestamp)
        return date_str

    def warn(self, msg):
        msg = "%s[WARN] %s" % (self.get_timestr(), msg)
        self.logger.warning(msg)
        self._write_file(msg)

    def info(self, msg):
        msg = "%s[INFO] %s" % (self.get_timestr(), msg)
        #self.logger.info(msg)
        self._write_file(msg)

    def error(self, msg):
        msg = "%s[ERROR] %s" % (self.get_timestr(), msg)
        self.logger.error(msg)
        self._write_file(msg)






class LHSampler:
    def __init__(self, sample_nums, sample_dim):
        self.min = -1.0
        self.max = 1.0
        self.samples = []
        self.is_init = False
        self.cursor = 0
        # self.sample_nums = 40
        # self.sample_dim = 0
        self.sample_nums = sample_nums
        self.sample_dim = sample_dim
        
        self.sample()
        
        
    def sample(self):
        
        rng = random.Random()
        
        for d in range(self.sample_dim):
            """
            灏嗕竴涓尯闂碵start, end]鍒嗘垚num_subintervals浠斤約1�7
            骞朵粠姣忎釜灏忓尯闂村唴闅忔溢�鍙栦竴涓€笺€ￄ1�7
            """
            subinterval_length = (self.max - self.min) / self.sample_nums
            single_dim_sample = [self.min + subinterval_length * i + rng.random() * subinterval_length for i in range(self.sample_nums)]
            tmp_single_dim_sample = [round(s, ndigits=8) for s in single_dim_sample]
            random.shuffle(tmp_single_dim_sample)
            self.samples.append(tmp_single_dim_sample)        



 
        print("\n=====================[LHS LOG]=====================")
        if len(self.samples) > 0:
            tmp_shape = (len(self.samples), len(self.samples[0]))
        else:
            tmp_shape = (len(self.samples), 0)
        
        print("LHS raw samples shape = {0}".format(tmp_shape))
        print("LHS raw samples = \n{0}".format(self.samples))
        print("=====================[LHS LOG]=====================\n")



    def resample(self):
        self.samples = []
        self.sample()        
    



    def get_sample_by_index(self, index):
        i = index % self.sample_nums
        res = [self.samples[j][i] for j in range(self.sample_dim)]
        print("\n=====================[LHS LOG]=====================")
        print("get sample by index = {0}: \n{1}".format(i, res))
        print("=====================[LHS LOG]=====================\n")
        return res
    
        
    def get_next_sample(self):
        res = self.get_sample_by_index(self.cursor)
        self.cursor += 1
        return res
        


def add_delta_ation_to_last_action(last_action, delta_action, ld_adaptor):
    print("[MERGE DEBUG] last_action = ", last_action)
    max_delta_action = globalValue.DELTA_ACTION_FOR_ADD
    delta_ld = delta_action
    print("[MERGE DEBUG] delta_ld = ", delta_ld)
    if ld_adaptor!=None:
        delta_hd = ld_adaptor.transform(delta_ld)
        print("[MERGE DEBUG] delta_hd = ", delta_hd)
    else:
        delta_hd = delta_ld
        print("[MERGE DEBUG] delta_hd = ", delta_hd)
    
    action = last_action + delta_hd * max_delta_action
    action = np.clip(action, -0.99999999, 0.99999999)
    print("[MERGE DEBUG] final action = ", action)
    return action


# def add_delta_ation_to_last_action_double(last_action, delta_action_1, ld_adaptor_1, delta_action_2, ld_adaptor_2):
#     print("[MERGE DEBUG] last_action = ", last_action)
    
    
#     max_delta_action = globalValue.DELTA_ACTION_FOR_ADD
    
#     delta_ld_1 = delta_action_1
#     print("[MERGE DEBUG] delta_ld_1 = ", delta_ld_1)
#     if ld_adaptor_1!=None:
#         delta_hd_1 = ld_adaptor_1.transform(delta_ld_1)
#         print("[MERGE DEBUG] delta_hd_1 = ", delta_hd_1)
#     else:
#         delta_hd_1 = delta_ld_1
#         print("[MERGE DEBUG] delta_hd_1 = ", delta_hd_1)
        
        
    
#     delta_ld_2 = delta_action_2
#     print("[MERGE DEBUG] delta_ld_2 = ", delta_ld_2)
#     if ld_adaptor_2!=None:
#         delta_hd_2 = ld_adaptor_2.transform(delta_ld_2)
#         print("[MERGE DEBUG] delta_hd_2 = ", delta_hd_2)
#     else:
#         delta_hd_2 = delta_ld_2
#         print("[MERGE DEBUG] delta_hd_2 = ", delta_hd_2)    
    
#     delta_hd = delta_hd_1 * globalValue.DOUBLE_RATIO + delta_hd_2 * (1.0 - globalValue.DOUBLE_RATIO)
    
#     action = last_action + delta_hd * max_delta_action
#     action = np.clip(action, -0.99999999, 0.99999999)
#     print("[MERGE DEBUG] final action = ", action)
#     return action
    


def add_delta_ation_to_last_action_double(last_action, delta_action_1, ld_adaptor_1, delta_action_2, ld_adaptor_2):
    print("[MERGE DEBUG] last_action = ", last_action)


    max_delta_action = globalValue.DELTA_ACTION_FOR_ADD

    delta_ld_1 = delta_action_1
    print("[MERGE DEBUG] delta_ld_1 = ", delta_ld_1)
    if ld_adaptor_1!=None:
        delta_hd_1 = ld_adaptor_1.transform(delta_ld_1)
        print("[MERGE DEBUG] delta_hd_1 = ", delta_hd_1)
    else:
        delta_hd_1 = delta_ld_1
        print("[MERGE DEBUG] delta_hd_1 = ", delta_hd_1)



    delta_ld_2 = delta_action_2
    print("[MERGE DEBUG] delta_ld_2 = ", delta_ld_2)
    if ld_adaptor_2!=None:
        delta_hd_2 = ld_adaptor_2.transform(delta_ld_2)
        print("[MERGE DEBUG] delta_hd_2 = ", delta_hd_2)
    else:
        delta_hd_2 = delta_ld_2
        print("[MERGE DEBUG] delta_hd_2 = ", delta_hd_2)

    delta_hd = delta_hd_1 * globalValue.DOUBLE_RATIO + delta_hd_2 * (1.0 - globalValue.DOUBLE_RATIO)

    delta_hd =  np.clip(delta_hd, -0.99999999, 0.99999999)


    action = last_action + delta_hd * max_delta_action * abs(last_action + 1.2)

    action = np.clip(action, -0.99999999, 0.99999999)
    print("[MERGE DEBUG] final action = ", action)
    return action





################################################
#############       娴嬭瘯閮ㄥ垎      ###############
###############################################


# from maEnv.env import SEEnv
#
if __name__ == '__main__':
    for i in range(10):
        rand = np.random.uniform(0, 1)
        print(rand)

    ########################鏁版嵁搴撹繛鎺ユ祴璇ￄ1�7###############################
    #conn = create_conn()
    #cur = get_cur(conn)

    # 鎵ц璇彄1�7
    #variables = ['innodb_buffer_pool_size', 'innodb_old_blocks_pct', 'innodb_old_blocks_time',
    #             'innodb_max_dirty_pages_pct_lwm', 'innodb_flush_neighbors', 'innodb_lru_scan_depth']
    #result = show_variables(cur,variables)
    #print(result)

    #close_conn_mysql(cur,conn)
    # variables = ['innodb_buffer_pool_size','innodb_old_blocks_pct','innodb_old_blocks_time','innodb_max_dirty_pages_pct_lwm','innodb_flush_neighbors','innodb_lru_scan_depth']
    #name = "innodb_buffer_pool_size"
    #value=134217728
    #conn_mysql(name,value)

    ########################鏁版嵁灏佽娴嬭瘄1�7###############################
    # new_variables = [1.0,2.0,4,3,10000]
    # set_variables_by_tune(new_variables)
    # print(s)
