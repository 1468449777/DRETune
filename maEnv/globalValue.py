#globalValue.py
# 主要配置文件，还有其他的一些配置分布在train_wxc.py和env.py
import threading
import os
# 线程交互事件
# EVENT = threading.Event()

# 参数配置信息
GLOBAL_ACTION = None

# status状态数组
GLOBAL_STATUS = []

# status状态数组的维度
# GLOBAL_STATUS_LIMIT = 5

# 当前指向status的位置
# GLOBAL_STATUS_POSITION = 0

# 最新的status
GLOBAL_CURRENT_STATUS = []

#innodb_buffer_pool_size大小
GLOBAL_BUFFER_POOL_SIZE = -1
GLOBAL_BUFFER_POOL_SIZE_CE = []
GLOBAL_BUFFER_POOL_SIZE_SE = []

#free链表的长度
# GLOBAL_FREE_LEN = 0

# chunk_size大小
# CHUNK_SIZE = 33554432
CHUNK_SIZE = 134217728

MAX_POOL_SIZE = 3355443200
# MAX_POOL_SIZE = pow(2, 64) - 1

MAX_POOL_LEN = GLOBAL_BUFFER_POOL_SIZE // 16 // 1024


# 每个episode开始时间
EPISODE_START_TIME = 0.0


#唤醒压测线程的事件
LOAD_EVENT = None


# 给SE判断episode是否结束的标志
SE_FLAG = False

EVAL_TEST = False

#GLOBAL_FULL_TAG = False

MAX_QPS = 0

MAX_HIT_RATIO = 0

SSH_CNT = 0

# 若要接入新系统新环境，你需要注意修改 
# 1. CONNECT_SE_IP, CONNECT_CE_IP
# 2. BASE_HOME
# 3. data_name
# 4. load_bash, load_tpcc
# 5. CE_LOAD_NUM

CONNECT_SE_IP = ['127.0.0.1']
CONNECT_CE_IP = ['127.0.0.1']   # primary ip 必须在index=0处

# 加了负载的CE的数量
CE_LOAD_NUM = 1

SE_PORT = 4000
CE_PORT = 2000
SSH_USERNAME = 'mxr'
SSH_PASSWD = '1'

# BASE_HOME = '/home/mxr/KnobsTuning/matune/MATune'


# 当前文件的上上级目录
BASE_HOME = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TMP_FILE = '/Users/mxr/Desktop/tmp/'

BEST_FILE = './test_model/'

ACTIONS_REWARD_FILE = "./test_model/actions_reward.csv"

BUFFER_POOL_SIZE_FILE = "./test_model/buffer_pool_size.txt"

CRITIC_LOSS_FILE = "./test_model/critic_loss.txt"

EVAL_REWARD_CAL_FILE = "./test_model/eval_reward_cal.txt"

HIT_RATIO_FILE = "./test_model/hit_ratio.txt"

QPS_STORE_FILE = "./test_model/qps_store.txt"

SCORES_FILE = "./test_model/scores.txt"

TRAIN_REWARD_CAL_FILE = "./test_model/train_reward_cal.txt"

RPM_SRC = f'{BASE_HOME}/matune_wxc/test_model/actions_reward.csv'
RPM_DEST = f'{BASE_HOME}/matune_wxc/test_model/actions_reward_new.csv'


MYSQLD_OPEN_EXEC_CE = f'nohup {BASE_HOME}/csdb_tune/csdb_buffer_tune/bld/sql/mysqld ' \
f'--ce=on --datadir={BASE_HOME}/csdb/ma/ScriptMA/ce_data/data '  \
f'--seuser=root --sepassword=root '  \
f'--innodb_use_native_aio=0 '  \
f'--configpath={BASE_HOME}/csdb_tune/csdb_buffer_tune/ma_se/ma_ce_config.json ' \
f'--skip-grant-tables --max_prepared_stmt_count=1000000 --max_connections=5000 ' \
f'--user=root '


MYSQLD_OPEN_EXEC_SE =f'{BASE_HOME}/csdb_tune/csdb_buffer_tune/bld/sql/mysqld ' \
f'--se=on --datadir={BASE_HOME}/csdb/ma/ScriptMA/se_data/data ' \
f'--seuser=root --sepassword=root ' \
f'--configpath={BASE_HOME}/csdb_tune/csdb_buffer_tune/ma_se/ma_se_config.json ' \
f'--innodb_use_native_aio=0 --skip-grant-tables ' \
f'--user=root '

# MYSQLD_OUTPUT = ' > /home/dawn/mysqld_output.txt 2>/home/dawn/mysqld_output.txt &'
MYSQLD_OUTPUT_CE = f' > {BASE_HOME}/csdb_tune/ce_output.txt 2>{BASE_HOME}/csdb_tune/ce_output.txt &'
MYSQLD_OUTPUT_SE = f' > {BASE_HOME}/csdb_tune/se_output.txt 2>{BASE_HOME}/csdb_tune/se_output.txt &'

# kill all mysqld
MYSQLD_CLOSE_EXEC = "ps aux|grep -v grep|grep mysqld|awk '{print $2}'|xargs kill -9"
MYSQLD_CE_CLOSE_EXEC = "ps aux|grep -v grep|grep mysqld|awk '$12 ~/ce/ {print $2}'|xargs kill -9"
MYSQLD_SE_CLOSE_EXEC = "ps aux|grep -v grep|grep mysqld|awk '$12 ~/se/ {print $2}'|xargs kill -9"

MYSQLD_CHECK = 'ps aux|grep -v grep|grep mysqld'


# 修改数据集需要配合压测指令一起修改
DATA_NAME = 'data_1GB.tar.gz'
NEW_DATA_CE_CMD = f''' \
echo "ready to prepare new data" && \
rm -rf {BASE_HOME}/csdb/ma/ScriptMA/ce_data/* && tar -xvf {BASE_HOME}/csdb/ma/ScriptMA/{DATA_NAME} -C {BASE_HOME}/csdb/ma/ScriptMA/ce_data/ > /dev/null && \
echo "finish" \
'''
NEW_DATA_SE_CMD = f''' \
echo "ready to prepare new data" && \
rm -rf {BASE_HOME}/csdb/ma/ScriptMA/se_data/* && tar -xvf {BASE_HOME}/csdb/ma/ScriptMA/{DATA_NAME} -C {BASE_HOME}/csdb/ma/ScriptMA/se_data/ > /dev/null && \
echo "finish" \
'''


# NEW_DATA_CE_CMD = f'{BASE_HOME}/csdb_tune/new_data_ce.sh'
# NEW_DATA_SE_CMD = f'{BASE_HOME}/csdb_tune/new_data_se.sh'
NEW_DATA_CE_TPCC_CMD = f'{BASE_HOME}/csdb_tune/new_data_ce_tpcc.sh'
NEW_DATA_SE_TPCC_CMD = f'{BASE_HOME}/csdb_tune/new_data_se_tpcc.sh'

# LOAD_BASH = 'nohup /home/dawn/test.sh ' \
#             '> /home/dawn/test_output.txt 2>/home/dawn/test_output.txt &'

LOAD_BASH = f'''nohup sysbench {BASE_HOME}/csdb_tune/sysbench-1.0.19/src/lua/oltp_read_write.lua --db-driver=mysql --mysql-user=root --mysql-password=mysql --mysql-host=localhost --mysql-port=3306 --mysql-db=test --mysql-storage-engine=innodb --mysql-socket=/tmp/mysql.sock --table-size=10000 --tables=100 --threads=128 --events=0 --report-interval=10 --range_selects=off --time=200  run \
            > {BASE_HOME}/csdb_tune/test_output.txt 2>{BASE_HOME}/csdb_tune/test_output.txt &'''

# LOAD_BASH = f'nohup sysbench {BASE_HOME}/csdb_tune/sysbench-1.0.19/src/lua/oltp_read_write.lua --db-driver=mysql --mysql-user=root --mysql-password=mysql --mysql-host=localhost --mysql-port=3306 --mysql-db=test --mysql-storage-engine=innodb --mysql-socket=/tmp/mysql.sock --table-size=10000 --tables=100 --threads=128 --events=0 --report-interval=10 --range_selects=off --time=200  run ' \
#             f'> {BASE_HOME}ma/test_output.txt 2>{BASE_HOME}ma/test_output.txt &'


# LOAD_BASH = f'nohup {BASE_HOME}/csdb_tune/test.sh ' \
#             f'> {BASE_HOME}/csdb_tune/test_output.txt 2>{BASE_HOME}//csdb_tune/test_output.txt &'

LOAD_TPCC = f'nohup {BASE_HOME}/csdb_tune/test_tpcc.sh ' \
            f'> {BASE_HOME}/csdb_tune/test_output.txt 2>{BASE_HOME}//csdb_tune/test_output.txt &'

TEST_RES_FILE = f'{BASE_HOME}/csdb_tune/test_output/'
TEST_TPCC_FILE = f'{BASE_HOME}/csdb_tune/tpcc_output/'
MAX_REWARD = -1

REWARD_NOW = -1




# 快速学习的早停阈值
FAST_LEARN_HIT_THRES = 0.7
# 生成状态时一个状态重复的个数（设为1就行）
FAST_LEARN_A_PER_S = 1#500
# 快速学习所用的随机数种子
FAST_LEARN_GEN_SEED = 35


# 双随机投影低维维度（第一个矩阵）
HES_LOW_ACT_DIM = 16
# 双随机投影低维维度（第二个矩阵）
HES_LOW_ACT_DIM_DOUBLE = 16
# 双随机投影加权权重
DOUBLE_RATIO = 0.4
# 双随机投影第一个矩阵生成的随机数种子
HES_LOW_SEED = 39
# 双随机投影第二个随机数种子与第一个随机数种子HES_LOW_SEED之间的差
DOUBLE_SEED_DELTA = 0
# grid search的范围
GRID_SERACH_AMPL = 5

# 是否开启proxysql
USE_PROXY_SQL = True

# 是否开启规则的SHAP分析
USE_SHAP_COLLECT_DATA = True

# 以前测试用的，已废弃
LOAD_TYPE = "wo"

# 关闭sysbench的指令
SYSBENCH_CLOSE = "pkill -9 sysbench"





# 是否使用部分规则
USE_PARTIAL_RULE = True
# 使用的部分规则的标号
PARTIAL_RULE = [2, 9, 11, 14, 19]
# 每次调优动作的步长
DELTA_ACTION_FOR_ADD = 0.2




# 
USE_MAKE_TRANSFER_DATA = True
TRANSFER_DATA_PATH = "wo"
TRANSFER_DATA_NUM = 10000






