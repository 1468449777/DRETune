from maEnv import utils


# send_variables_se = "3$se$17$buffer_pool_size$134217728$old_blocks_pct$37$old_threshold_ms$1000$flush_neighbors$1$lru_sleep_time_flush$1000$flush_n$1024$SE_LRU_idle_scan_depth$20$lru_scan_depth$1024$lru_sleep_time_remove$1000$reserve_free_page_pct_for_se$50$free_page_threshold$8192$max_dirty_pages_pct_lwm$0$adaptive_flushing_lwm$10$flushing_avg_loops$30$random_read_ahead$0$read_ahead_threshold$56$io_capacity$200"

# ip_se = "127.0.0.1"
# port_se = 4000

# utils.send_msg_to_server(send_variables_se, ip_se, port_se)


send_variables_ce = "3$ce$8$buffer_pool_size$188888888$old_blocks_pct$50$old_threshold_ms$900$ce_coordinator_sleep_time$1024$ce_free_page_threshold$8192$random_read_ahead$0$read_ahead_threshold$56$flushing_avg_loops$30"
ip_ce = "127.0.0.1"
port_ce = 2000

utils.send_msg_to_server(send_variables_ce, ip_ce, port_ce)
bps_ce = utils.get_bps(ip_ce, port_ce)

print("ce bps = ", bps_ce)


