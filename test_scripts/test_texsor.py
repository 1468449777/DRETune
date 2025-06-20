# import socket
# import time

# if __name__ == '__main__':
#     p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     p.connect(('127.0.0.1', 30001))

#     msg1 = '3$ce$100000000$50$50$50$50'
#     p.send(msg1.encode('utf-8'))
#     received_msg = p.recv(1024)
#     print('received_msg:',received_msg.decode('utf-8'))


#     # time.sleep(1000)
#     msg_exit = 'exit'
#     p.send(msg_exit.encode('utf-8'))
#     received_msg = p.recv(1024)
#     print('received_msg:', received_msg.decode('utf-8'))

#     p.close()

import paddle.fluid as fluid
import threading
import paddle
print(paddle.__version__)


with fluid.dygraph.guard():

    def model_operation():
        # paddle.disable_static()  # 确保在函数开始处调用
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        y = paddle.to_tensor([4.0, 5.0, 6.0])
        z = x + y
        print("Result in thread:", z.numpy())

    # 创建并启动线程
    thread = threading.Thread(target=model_operation)
    thread.start()
    thread.join()


[
    ('0#buffer_pool_size', 0.115),
    ('0#old_threshold_ms', 0.111),
    ('0#lru_sleep_time_flush', 0.094),
    ('0#SE_LRU_idle_scan_depth', 0.062),
    ('1#buffer_pool_size', 0.059),
    ('0#free_page_threshold', 0.058),
    ('0#read_ahead_threshold', 0.054),
    ('0#reserve_free_page_pct_for_se', 0.051),
    ('1#ce_free_page_threshold', 0.047),
    ('0#lru_sleep_time_remove', 0.043),
    ('1#old_threshold_ms', 0.037),
    ('0#max_dirty_pages_pct_lwm', 0.034),
    ('0#lru_scan_depth', 0.033),
    ('1#read_ahead_threshold', 0.031),
    ('1#old_blocks_pct', 0.028),
    ('1#ce_coordinator_sleep_time', 0.027),
    ('0#io_capacity', 0.025),
    ('0#flushing_avg_loops', 0.023),
    ('0#old_blocks_pct', 0.022),
    ('0#flush_n', 0.022),
    ('0#adaptive_flushing_lwm', 0.016),
    ('1#flushing_avg_loops', 0.005),
    ('0#flush_neighbors', 0.003),
    ('0#random_read_ahead', 0.0),
    ('1#random_read_ahead', 0.0)
][
    ('1#buffer_pool_size', 0.204),
    ('1#ce_free_page_threshold', 0.145),
    ('0#read_ahead_threshold', 0.075),
    ('0#lru_scan_depth', 0.054),
    ('0#reserve_free_page_pct_for_se', 0.05),
    ('0#lru_sleep_time_flush', 0.049),
    ('0#buffer_pool_size', 0.047),
    ('1#read_ahead_threshold', 0.043),
    ('0#old_threshold_ms', 0.04),
    ('0#SE_LRU_idle_scan_depth', 0.036),
    ('0#flushing_avg_loops', 0.036),
    ('1#old_threshold_ms', 0.035),
    ('0#io_capacity', 0.032),
    ('1#old_blocks_pct', 0.029),
    ('0#max_dirty_pages_pct_lwm', 0.023),
    ('1#flushing_avg_loops', 0.021),
    ('0#old_blocks_pct', 0.018),
    ('0#adaptive_flushing_lwm', 0.018),
    ('1#ce_coordinator_sleep_time', 0.018),
    ('0#flush_n', 0.015),
    ('0#free_page_threshold', 0.008),
    ('0#lru_sleep_time_remove', 0.005),
    ('0#flush_neighbors', 0.0),
    ('0#random_read_ahead', 0.0),
    ('1#random_read_ahead', 0.0)
]


[
    ('0#old_blocks_pct', 0.115),
    ('0#free_page_threshold', 0.113),
    ('1#flushing_avg_loops', 0.078),
    ('0#lru_sleep_time_flush', 0.075),
    ('0#lru_sleep_time_remove', 0.074),
    ('0#SE_LRU_idle_scan_depth', 0.065),
    ('0#buffer_pool_size', 0.064),
    ('0#old_threshold_ms', 0.061),
    ('0#io_capacity', 0.05),
    ('0#adaptive_flushing_lwm', 0.047),
    ('1#read_ahead_threshold', 0.044),
    ('1#old_blocks_pct', 0.031),
    ('1#old_threshold_ms', 0.03),
    ('1#buffer_pool_size', 0.029),
    ('0#lru_scan_depth', 0.027),
    ('0#max_dirty_pages_pct_lwm', 0.027),
    ('0#flush_n', 0.022),
    ('1#ce_free_page_threshold', 0.019),
    ('0#reserve_free_page_pct_for_se', 0.014),
    ('0#read_ahead_threshold', 0.01),
    ('1#ce_coordinator_sleep_time', 0.005),
    ('0#flush_neighbors', 0.0),
    ('0#flushing_avg_loops', 0.0),
    ('0#random_read_ahead', 0.0),
    ('1#random_read_ahead', 0.0)
]
