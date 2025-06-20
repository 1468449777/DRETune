from maEnv import utils
from maEnv import datautils
from maEnv import env
import time
import sys
from venv import logger
import paramiko as paramiko
import time
import globalValue 
import socket
import threading
import pymysql


LOAD_WAIT_TIME = 100
LOAD_COLLECT_TIME = 30
BEFORE_LOAD_BASH_TIME = 5

# DEFAULT_ACTION = {
#     "se1": [134217728, 37, 1000, 1, 1000, 1024, 20, 1024, 1000, 50, 8192, 0, 10, 30, 0, 56, 200],
#     "ce1": [134217728, 37, 1000, 1024, 8192, 0, 56, 30]
# }

def create_conn(ip) -> object:

    #连接配置信息
    # conn_host='192.168.1.102'
    # conn_host='127.0.0.1'
    conn_host=ip
    conn_port=3306
    conn_user='root'
    conn_password='mysql'

    #建立连接
    conn = pymysql.connect(host=conn_host, port=conn_port, user=conn_user, password=conn_password)
    return conn

# 获取数据库最新state
def get_current_status():
    if(len(globalValue.GLOBAL_CURRENT_STATUS) == 0):
        return None
    return globalValue.GLOBAL_CURRENT_STATUS

# 设置数据库最新状态
def set_curent_status(current_status):
    globalValue.GLOBAL_CURRENT_STATUS = current_status

# 建立数据库连接
def get_cur(conn):
    cur = conn.cursor()
    #print('Conntion created successfully!!!')
    return cur

#关闭数据库连接
def close_conn_mysql(cur, conn):
    cur.close()
    conn.close()
    #print('Conntion closed successfully!!!')




class NodeState():
    def __init__(self, name, ip, port, uuid):
        self.name = name
        self.ip = ip
        self.port = port
        # 唯一区分表示
        self.uuid = uuid
        
        
        
se_info = []
ce_info = []

for i in range(len(globalValue.CONNECT_SE_IP)):
    se_node = NodeState("se"+str(i+1), globalValue.CONNECT_SE_IP[i], globalValue.SE_PORT, str(i+1))
    se_info.append(se_node)
    
for i in range(len(globalValue.CONNECT_CE_IP)):
    ce_node = NodeState("ce"+str(i+1), globalValue.CONNECT_CE_IP[i], globalValue.CE_PORT, str(i+1))
    ce_info.append(ce_node)
    

# se_1 = NodeState("se", globalValue.CONNECT_SE_IP[0], globalValue.SE_PORT, '1')
# ce_1 = NodeState("ce", globalValue.CONNECT_CE_IP[0], globalValue.CE_PORT, '2')

# ce_info = [
#     ce_1, 
# ]

# se_info = [se_1]

se_var_names = {
    # se
    'buffer_pool_size': [134217728, 33554432, 3355443200],
    'old_blocks_pct': [37, 5, 95],
    'old_threshold_ms': [1000, 0, 10000],
    'flush_neighbors': [1, 0, 2],
    'lru_sleep_time_flush': [1000, 50, 2000],
    'flush_n': [1024, 0, 3200],
    'SE_LRU_idle_scan_depth': [20, 0, 100],
    'lru_scan_depth': [1024, 1024, 10240],
    'lru_sleep_time_remove': [1000, 0, 1000],
    'reserve_free_page_pct_for_se': [50, 5, 95],
    'free_page_threshold': [8192, 1024, 10240],  
    'max_dirty_pages_pct_lwm': [0, 0, 99.99],
    'adaptive_flushing_lwm': [10, 0, 70],
    'flushing_avg_loops': [30, 1, 1000],
    'random_read_ahead': [0, 0, 1],
    'read_ahead_threshold': [56,0,64],
    'io_capacity': [200, 100, 10000],
}


ce_var_names = {
    # ce 
    'buffer_pool_size': [134217728, 33554432, 3355443200],
    'old_blocks_pct': [37, 5, 95],
    'old_threshold_ms': [1000, 0, 10000],
    'ce_coordinator_sleep_time': [1024, 50, 2000],
    'ce_free_page_threshold': [8192, 1024, 10240],
    'random_read_ahead': [0, 0, 1],
    'read_ahead_threshold': [56,0,64],
    'flushing_avg_loops': [30, 1, 1000],
}


DEFAULT_ACTION = {}
se_act = {}
for key in se_var_names.keys():
    # print("[DEBUG][DEFAULT ACTION] key = {0}".format(key))
    se_act[key] = se_var_names[key][0]
    # print("[DEBUG][DEFAULT ACTION] se_act = {0}".format(se_act))
DEFAULT_ACTION["se1"] = se_act

ce_act = {}
for key in ce_var_names.keys():
    ce_act[key] = ce_var_names[key][0]
DEFAULT_ACTION["ce1"] = ce_act

# print("[DEBUG][DEFAULT ACTION] DEFAULT_ACTION = {0}".format(DEFAULT_ACTION))








# 将给定字符串发送给数据库
def send_msg_to_server(msg, ip, port):
    p = None
    received_msg = None
    # print('send message-> : ', msg)
    try:
        # 建立连接
        # print('msg = {} ip = {} port = {}'.format(msg, ip, port))
        p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        p.connect((ip, port))
        # 发送消息
        p.send(msg.encode('utf-8'))
        # 接收反馈
        received_msg = p.recv(1024).decode('utf-8')
        # print('received_msg = ',received_msg)
        p.send("exit".encode('utf-8'))
    except Exception as e:
        print("[{0}][{1}] {2}".format(ip, port, e))
        # print('Meet set variables failure!!')
        p.close()
        return received_msg, False
    # 关闭连接
    p.close()
    return received_msg, True


# 远程连接linux, 并实现远程启动数据库
def sshExe(sys_ip,username,password,cmd):
    client = None
    result = None
    # print(cmd)
    try:
        #创建ssh客户端
        client = paramiko.SSHClient()
        #第一次ssh远程时会提示输入yes或者no
        # if globalValue.SSH_CNT <= 5:
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        #密码方式远程连接
        client.connect(sys_ip, 22, username=username, password=password, timeout=20)
        #互信方式远程连接
        #key_file = paramiko.RSAKey.from_private_key_file("/root/.ssh/id_rsa")
        #ssh.connect(sys_ip, 22, username=username, pkey=key_file, timeout=20)
        #执行命令
        # stdin, stdout, stderr = client.exec_command(cmds)
        stdin, stdout, stderr = client.exec_command(cmd)
        #获取命令执行结果,返回的数据是一个list
        result = stdout.readlines()
    except Exception as e:
        print("[{0}] {1}".format(sys_ip, e))
    finally:
        client.close()
        return result

# 远程开启负载压测
def load_bash_remote(type):
    # load 80s
    # print('LOAD BASH++++')
    if type == 'sysbench':
        cmd = globalValue.LOAD_BASH
        # print(cmd)
        flag = list()
        for i in range(globalValue.CE_LOAD_NUM):
            tmp_flag = sshExe(globalValue.CONNECT_CE_IP[i], globalValue.SSH_USERNAME, globalValue.SSH_PASSWD, cmd)
            flag = flag.append(tmp_flag)
    elif type == 'tpcc':
        cmd = globalValue.LOAD_TPCC
        # prepare_for_tpcc(globalValue.CONNECT_CE_IP[0])
        # print(cmd)
        flag = sshExe(globalValue.CONNECT_CE_IP[0], globalValue.SSH_USERNAME, globalValue.SSH_PASSWD, cmd)
    return flag


def start_for_a_not_fresh_node(node_close_cmd, node_start_cmd):
    # sleep
    sleep_cmd = 'sleep 1;'
    # close
    close_node = node_close_cmd + ';'
    # start with exact string
    start_node = node_start_cmd
    text = close_node + sleep_cmd + start_node
    return text

def get_set_variables_string(var_names, new_values, node_name, type):
    length = len(var_names)
    s = str(type) + '$'
    if 'se' in node_name:
        s = s + 'se'
    else:
        s = s + 'ce'
    # 封装参数个数
    s = s + '$' + str(length)
    # 增加参数姓名的封装
    for i in range(length):
        # print("[DEBUG] var_names[i] = {0}, new_values = {1}, node_name = {2}".format(var_names[i], new_values, node_name))
        s = s + '$' + var_names[i] + '$' + str(new_values[var_names[i]])
    print(s)
    return s

# 启动某节点
BASH_TYPE='sysbench'
# real action 就是你要设置参数的值，顺序按照se_var_names和ce_var_names中来
def connect_with_node(node, real_action):

    if node.name.startswith('se'):
        buffer_set = ' --innodb_buffer_pool_size=' + str(real_action['buffer_pool_size'])
    else:
        buffer_set = ' --innodb_buffer_pool_size=' + str(real_action['buffer_pool_size'])
    
    if 'se' in node.name:
        start_cmd = globalValue.MYSQLD_OPEN_EXEC_SE + "--sehost=%"  + buffer_set + globalValue.MYSQLD_OUTPUT_SE
        close_cmd = globalValue.MYSQLD_SE_CLOSE_EXEC
        if BASH_TYPE == 'sysbench':
            new_data_cmd = globalValue.NEW_DATA_SE_CMD
        else:
            new_data_cmd = globalValue.NEW_DATA_SE_TPCC_CMD

    else:
        if node.ip!=globalValue.CONNECT_SE_IP[0]:
            start_cmd = globalValue.MYSQLD_OPEN_EXEC_CE + "--sehost=" + globalValue.CONNECT_SE_IP[0] + buffer_set + globalValue.MYSQLD_OUTPUT_CE
        else:
            start_cmd = globalValue.MYSQLD_OPEN_EXEC_CE + "--sehost=127.0.0.1" + buffer_set + globalValue.MYSQLD_OUTPUT_CE
        close_cmd = globalValue.MYSQLD_CE_CLOSE_EXEC
        if BASH_TYPE == 'sysbench':
            new_data_cmd = globalValue.NEW_DATA_CE_CMD
        else:
            new_data_cmd = globalValue.NEW_DATA_CE_TPCC_CMD


    print("\nnode_name=" + node.name + "\t" + node.ip)
    print(start_cmd + "\n")

    # sshExe(node.ip, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD, new_data_cmd)
    # time.sleep(2)
    text = start_for_a_not_fresh_node(close_cmd, start_cmd)
    text = new_data_cmd + ";" + text
    # start node
    sshExe(node.ip, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD, text)
    
    print('--->1.启动%s  %s' % (node.name, node.uuid))
    time.sleep(5)
    # 将推荐参数封装为发送数据标准格式
    if 'se' in node.name:
        var_names = list(se_var_names.keys())
    else:
        var_names = list(ce_var_names.keys())
    
    send_variables = get_set_variables_string(var_names, real_action, node.name, 3)
    # 动作变量默认值设置，此时需要确认是否成功连接mysql
    received_msg, flag = send_msg_to_server(send_variables, node.ip, node.port)
    return received_msg, flag, send_variables
    



def close_sysbench():
    for i in range(1):
            tmp_flag = sshExe(globalValue.CONNECT_CE_IP[i], globalValue.SSH_USERNAME, globalValue.SSH_PASSWD, globalValue.SYSBENCH_CLOSE)
    time.sleep(1)

# 启动所有节点
def start_all_nodes(real_action, ses_send, ces_send):
        # print("[DEBUG][start_all_nodes] real_action = {0}".format(real_action))
        flag = True
        flag_se = True
        flag_ce = True
        
        # 需要考虑启动失败的情况（异常处理）
        reset_times = 0

        while True:
            close_all_nodes()
            reset_times += 1
            if reset_times == 3:
                flag = False
                break
            # 依次启动所有se和ce
            
            
            def connect_with_se(node , flag_se):
                received_msg_se, flag_se, send_variables_se = connect_with_node(node, real_action[node.name])
                if flag_se == False:
                    return
                time.sleep(2)
            
            ts_se = []
            for se in se_info:
                t = threading.Thread(target=connect_with_se, args=(se, flag_se))
                ts_se.append(t)
            for t in ts_se:
                t.start()
            for t in ts_se:
                t.join()
            
            

            if flag_se == False:
                continue

            
            def connect_with_ce(node, flag_ce):
                received_msg_ce, flag_ce, send_variables_ce = connect_with_node(node, real_action[node.name])
                if flag_ce == False:
                    time.sleep(5)
                    received_msg_ce, flag_ce, send_variables_ce = connect_with_node(node, real_action[node.name])
                    if flag_ce == False:
                        return
                time.sleep(2)

            ts_ce = []
            for idx, ce in enumerate(ce_info):
                if idx == 0:
                    connect_with_ce(ce, flag_ce)
                else:
                    t = threading.Thread(target=connect_with_ce, args=(ce, flag_ce))
                    ts_ce.append(t)
                
            for t in ts_ce:
                t.start()
                time.sleep(10)
            for t in ts_ce:
                t.join()
                

            if flag_ce == False:
                continue
            else:
                break

        return flag



def close_all_nodes():
    for ce in ce_info:
        sshExe(ce.ip, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD,
                        globalValue.MYSQLD_CE_CLOSE_EXEC)
    for se in se_info:
        sshExe(se.ip, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD,
                        globalValue.MYSQLD_SE_CLOSE_EXEC)

# 获取状态变量当前值,show global status like...
# 输入为单个状态变量
# [questions]用来计算reward
def show_status(cur,status_name):
    # 待执行语句
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
    
def get_qps(node):
    conn = create_conn(node.ip)
    cur = get_cur(conn)
    p1 = show_status(cur, 'questions')
    time.sleep(30)
    p2 = show_status(cur, 'questions')
    p_1 = (int(p2[0][1]) - int(p1[0][1])) / 30.0


    # print('The current qps is:',p)
    close_conn_mysql(cur, conn)
    return p_1

# 使用sql语句计算hit_ratio
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
    
# 计算hit_ratio
def get_se_hr(node):
    ip = node.ip
    port = node.port
    # 获取当前页面总数
    pages_info_before, flag = send_msg_to_server('2', ip, port)
    # 计算5秒的hit_ratio
    time.sleep(5)
    # 获取当前页面总数
    pages_info_after, flag = send_msg_to_server('2',ip,port)

    hr = cal_hr(pages_info_before, pages_info_after)
    return hr


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


def get_bps(ip, port):
    bps, flag = send_msg_to_server('4', ip, port)
    print("[{0}:{1}] bps = {2}".format(ip, port, bps))
    bps = int(bps.split("$")[0])
    return bps


def get_all_nodes_state():
        current_nodes_state = []
        current_nodes_state_list = []
        for se in se_info:
            current_state_se, flag1 = utils.send_msg_to_server('1', se.ip, se.port)
            # print("current state se->: ", current_state_se)
            current_state_se = datautils.GetNodeState(current_state_se, se.name)
            se.state_now = current_state_se
            current_nodes_state += current_state_se
            

        for ce in ce_info:
            current_state_ce, flag1 = utils.send_msg_to_server('1', ce.ip, ce.port)
            # print("current state ce->: ", current_state_ce)
            current_state_ce = datautils.GetNodeState(current_state_ce, ce.name)
            ce.state_now = current_state_ce
            current_nodes_state += current_state_ce

        print("[NOTION] The current_state is : ", current_nodes_state)
        return current_nodes_state


# def connect_socket(msg, ip, port):
#     p = None
#     received_msg = None
#     try:
#         # 建立连接
#         # print('msg = {} ip = {} port = {}'.format(msg, ip, port))
#         p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         p.connect((ip, port))
#         # 发送消息
#         p.send(msg.encode('utf-8'))
#         # 接收反馈
#         received_msg = p.recv(1024).decode('utf-8')
#         print('received_msg = ',received_msg)
#         p.send("exit".encode('utf-8'))
#     except Exception as e:
#         print(e)
#         print('connect fail !!')
#         p.close()
#     return received_msg
        

# def connect_test():
#     msg = "1"
#     ip = "127.0.0.1"
#     port = 5786
#     connect_socket(msg, ip, port)

def make_load_feature_vec():
    msg = "2$" + str(LOAD_COLLECT_TIME) + "$"
    ip = "127.0.0.1"
    port = 5786
    
    
    print("[LOG] begin to start all nodes ...")
    flag = False
    while not flag:
        flag = start_all_nodes(DEFAULT_ACTION, {}, {})
        print("[NOTION] start flag = {0}".format(flag))
    
    close_sysbench()
    print("[LOG] waiting {0} s for bash ...".format(BEFORE_LOAD_BASH_TIME))
    time.sleep(BEFORE_LOAD_BASH_TIME)
    print("[LOG] begin to load bash ...")
    load_bash_remote('sysbench')

    print("[LOG] wait load for {0} s ...".format(LOAD_WAIT_TIME))
    time.sleep(LOAD_WAIT_TIME)
    
    print("[LOG] begin to collect wait for {0} s ...".format(LOAD_COLLECT_TIME))
    collect_load_data, done = utils.send_msg_to_server(msg, ip, port) 
    if done:
        print("[NOTION] collect_load_data = {0}".format(collect_load_data))
        
    print("[LOG] begin to get all nodes status ...")
    current_nodes_state = get_all_nodes_state()
    
    print("[NOTION] current_nodes_state = {0}".format(current_nodes_state))
    
    load_feature_vec = []
    # compact collect_load_data and status
    load_list = collect_load_data.split("$")
    print("[NOTION] load_list len = {0}".format(len(load_list)))
    # print("[NOTION] load_data_value_list = {0}".format(load_list))
    for i in range(len(load_list) - 1):
        load_feature_vec.append((float)(load_list[i]))
    load_feature_vec += current_nodes_state
    print("[NOTION] load_feature_vec = {0}".format(load_feature_vec))
    return load_feature_vec
    
    

def normalize_state_vec(nodes_state, mode = 2):
    
    crud_range = [
        ["int", 0, 10000],          # 0     free_list_len
        ["int", 0, 10000],          # 1     lru_len
        ["int", 0, 10000],          # 2     old_lru_len
        ["int", 0, 10000],          # 3     flush_list_l
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
    
        print("before normalize: {0}".format(nodes_state))
        node_obs_dim = 14
        for idx, s in enumerate(nodes_state):
            mod = idx % node_obs_dim
            nodes_state[idx] = (float)(nodes_state[idx] - knobs_range[mod][1]) / (float)(knobs_range[mod][2] - knobs_range[mod][1])
        
        print("after  normalize: {0}".format(nodes_state))
        return nodes_state
    elif mode == 2:
        part_node_state = nodes_state[22:]
        print("before normalize(after extract): {0}".format(part_node_state))
        print("before normalize(after extract) len = : {0}".format(len(part_node_state)))
        node_obs_dim = 14
        for idx, s in enumerate(part_node_state):
            mod = idx % node_obs_dim
            part_node_state[idx] = (float)(part_node_state[idx] - knobs_range[mod][1]) / (float)(knobs_range[mod][2] - knobs_range[mod][1])
        
        print("after  normalize(after extract): {0}".format(part_node_state))
        
        for idx, s in enumerate(nodes_state):
            if idx >= 22:
                nodes_state[idx] = part_node_state[idx-22]
        print("after  normalize(all   states ): {0}".format(nodes_state))
        
        
        part_node_state = nodes_state[0:4]
        print("before normalize(after extract): {0}".format(part_node_state))
        print("before normalize(after extract) len = : {0}".format(len(part_node_state)))
        node_obs_dim = 4
        for idx, s in enumerate(part_node_state):
            mod = idx % node_obs_dim
            part_node_state[idx] = (float)(part_node_state[idx] - crud_range[mod][1]) / (float)(crud_range[mod][2] - crud_range[mod][1])
        
        print("after  normalize(after extract): {0}".format(part_node_state))
        
        for idx, s in enumerate(nodes_state):
            if idx < 4:
                nodes_state[idx] = part_node_state[idx]
        print("after  normalize(all   states ): {0}".format(nodes_state))
        
        
        return nodes_state
        
    
# make_load_feature_vec()
    
# connect_test()
        
# collect_test()  




import numpy as np
from scipy.stats import pearsonr

def standardize_and_compute_similarity(vec1, vec2):
    """
    标准化两个负载特征向量，并计算它们之间的皮尔逊相关系数（相似度）。
    
    参数:
    vec1 (list): 第一个负载特征向量
    vec2 (list): 第二个负载特征向量
    
    返回:
    float: 皮尔逊相关系数，表示两个向量之间的相似度
    """
    # 将向量转化为 NumPy 数组
    all_vectors = np.array([vec1, vec2])
    
    # 计算每个特征的均值和标准差
    means = np.mean(all_vectors, axis=0)  # 计算每列（特征）的均值
    stds = np.std(all_vectors, axis=0)    # 计算每列（特征）的标准差
    
    # 处理标准差为零的情况，避免除以零
    stds = np.where(stds == 0, 1.0, stds)  # 将标准差为零的地方替换为1，避免除以零
    
    # 标准化数据
    normalized_vectors = (all_vectors - means) / stds
    
    # 提取标准化后的两个向量
    normalized_vec1 = vec1 # normalized_vectors[0]
    normalized_vec2 = vec2 # normalized_vectors[1]
    # print("[NOTION] normalized_vec1 = {0}".format(normalized_vec1))
    # print("[NOTION] normalized_vec2 = {0}".format(normalized_vec2))
    
    # 计算皮尔逊相关系数
    correlation, _ = pearsonr(normalized_vec1, normalized_vec2)
    # print(f"[NOTION] 皮尔逊相关系数: {correlation}")
    
    return correlation

# 示例数据
load_feature_vec_rw_3_2 = [69640.0, 6964.0, 13928.0, 6964.0, -0.007617, -0.007617, -0.015234, -0.07617, -0.030468, -0.106638, -0.007617, -0.030468, -0.015234, -0.083787, -0.007617, -0.099021, -0.007617, -0.007617, -0.007617, -0.007617, -0.099021, -0.07617, 31629, 1138, 440, 256, 0, 0, 0, 322, 15462, 158472, 4.11065, 0.0, 308.632, 0.0, 6895, 1159, 447, 700, 0, 0, 0, 0, 57496, 2922070, 13.5206, 4.77073, 0.0, 0.0]
load_feature_vec_test = [9640.0, 964.0, 13928.0, 764.0, -0.007617, -0.007617, -0.015234, -0.07617, -0.030468, -0.106638, -0.007617, -0.030468, -0.015234, -0.083787, -0.007617, -0.099021, -0.007617, -0.007617, -0.007617, -0.007617, -0.099021, -0.07617, 31629, 1138, 440, 256, 0, 0, 0, 322, 15462, 158472, 4.11065, 0.0, 308.632, 0.0, 6895, 1159, 447, 700, 0, 0, 0, 0, 57496, 2922070, 13.5206, 4.77073, 0.0, 0.0]
load_feature_vec_ro = [177617.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.111798, 0.0, -0.111798, 0.0, 0.0, 0.0, -0.111798, -0.01118, -0.111798, 0.0, -0.01118, 0.0, 0.0, -0.111798, -0.111798, 32320, 448, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 7595, 459, 0, 37, 0, 0, 0, 0, 25, 1444783, 3.68742, 0.0208329, 0.0, 0.0]
load_feature_vec_wo = [0.0, 12221.0, 24441.0, 12220.0, -0.02236, -0.022358, -0.044719, 0.0, -0.089441, -0.089437, -0.02236, -0.089441, -0.044719, -0.022358, -0.02236, -0.067077, -0.02236, -0.02236, -0.02236, -0.022358, -0.067077, 0.0, 31558, 1206, 465, 323, 0, 0, 0, 0, 199, 1950, 0.0, 0.0, 0.0, 0.0, 6826, 1227, 472, 703, 0, 0, 0, 0, 103056, 4183160, 14.5507, 4.755, 0.0, 0.0]

# print(len(load_feature_vec_ro))
# load_feature_vec_ro = normalize_state_vec(load_feature_vec_ro, mode = 2)
# load_feature_vec_wo = normalize_state_vec(load_feature_vec_wo, mode = 2)
# load_feature_vec_test = normalize_state_vec(load_feature_vec_test, mode = 2)
# load_feature_vec_rw_3_2 = normalize_state_vec(load_feature_vec_rw_3_2, mode = 2)
# similarity = standardize_and_compute_similarity(load_feature_vec_ro, load_feature_vec_test)
# print("load_feature_vec_ro * load_feature_vec_ro = ", similarity)
# similarity = standardize_and_compute_similarity(load_feature_vec_wo, load_feature_vec_test)
# print("load_feature_vec_wo * load_feature_vec_ro = ", similarity)
# similarity = standardize_and_compute_similarity(load_feature_vec_rw_3_2, load_feature_vec_test)
# print("load_feature_vec_rw_3_2 * load_feature_vec_ro = ", similarity)


# # 调用函数计算相似度
# # similarity = standardize_and_compute_similarity(load_feature_vec1, load_feature_vec2)
# # make_load_feature_vec()



        
