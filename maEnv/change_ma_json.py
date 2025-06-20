import json
import paramiko
from maEnv import globalValue
def update_ce_remote_json_file(ssh_host, ssh_user, ssh_password, json_path,ce_idx):
    """
    远程更新 JSON 文件中 node0 的 IP 地址。

    :param ssh_host: 远程主机 IP 地址
    :param ssh_user: SSH 用户名
    :param ssh_password: SSH 密码
    :param json_path: 远程 JSON 文件路径（例如：/base/home/ma_ce.json）
    :param new_ip: 要更新为的新 IP 地址
    """
    try:
        # 创建 SSH 客户端
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(ssh_host, username=ssh_user, password=ssh_password)

        # 读取远程 JSON 文件内容
        sftp = ssh.open_sftp()
        with sftp.open(json_path, 'r') as f:
            json_data = json.load(f)  # 加载 JSON 内容为字典对象

        # 修改 JSON 中的 SE IP,目前默认只有一个存储节点，若是多个存储节点，需要修改
        if "qg_protocol" in json_data and "node0" in json_data["qg_protocol"]:
            json_data["qg_protocol"]["node0"]["ip"] = globalValue.CONNECT_SE_IP[0]
            print(f"Updated 'SE_ip' IP to: {globalValue.CONNECT_SE_IP[0]}")
        else:
            print("Error: Target key 'qg_protocol.node0.ip' not found in JSON file.")
            return
        
        # 修改 JSON 中的 CE IP
        if "master" in json_data and "ip" in json_data["master"]:
            json_data["master"]["ip"] = globalValue.CONNECT_CE_IP[0]
            print(f"Updated 'ce_master_ip' IP to: {globalValue.CONNECT_CE_IP[0]}")
            if "srv_is_master" in json_data:
                if ce_idx == 0:
                    json_data["srv_is_master"] = 1
                else:
                    json_data["srv_is_master"] = 0
            else:
                print("Error: Target key 'srv_is_master' not found in JSON file.")
            
        else:
            print("Error: Target key 'master.ip' not found in JSON file.")
            return        

        # 将修改后的 JSON 写回远程文件
        with sftp.open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
            print(f"Successfully updated JSON file at: {json_path}")

        # 关闭 SFTP 和 SSH 连接
        sftp.close()
        ssh.close()

    except Exception as e:
        print(f"Error occurred: {e}")

def update_se_remote_json_file(ssh_host, ssh_user, ssh_password, json_path,ce_idx):
    """
    远程更新 JSON 文件中 node0 的 IP 地址。

    :param ssh_host: 远程主机 IP 地址
    :param ssh_user: SSH 用户名
    :param ssh_password: SSH 密码
    :param json_path: 远程 JSON 文件路径（例如：/base/home/ma_ce.json）
    :param new_ip: 要更新为的新 IP 地址
    """
  
    try:
        # 创建 SSH 客户端
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(ssh_host, username=ssh_user, password=ssh_password)

        # 读取远程 JSON 文件内容
        sftp = ssh.open_sftp()
        with sftp.open(json_path, 'r') as f:
            json_data = json.load(f)  # 加载 JSON 内容为字典对象

        # 修改 JSON 中的 SE IP,目前默认只有一个存储节点，若是多个存储节点，需要修改
        if "qg_protocol" in json_data and "node0" in json_data["qg_protocol"]:
            json_data["qg_protocol"]["node0"]["ip"] = globalValue.CONNECT_SE_IP[0]
            print(f"Updated 'SE_ip' IP to: {globalValue.CONNECT_SE_IP[0]}")
        else:
            print("Error: Target key 'qg_protocol.node0.ip' not found in JSON file.")
            return
        
        # 修改 JSON 中的 CE IP
        if "master" in json_data and "ip" in json_data["master"]:
            json_data["master"]["ip"] = globalValue.CONNECT_CE_IP[0]
            print(f"Updated 'ce_master_ip' IP to: {globalValue.CONNECT_CE_IP[0]}")
            if "srv_is_master" in json_data:
                json_data["srv_is_master"] = 1
            else:
                print("Error: Target key 'srv_is_master' not found in JSON file.")
            
        else:
            print("Error: Target key 'master.ip' not found in JSON file.")
            return        

        # 将修改后的 JSON 写回远程文件
        with sftp.open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
            print(f"Successfully updated JSON file at: {json_path}")

        # 关闭 SFTP 和 SSH 连接
        sftp.close()
        ssh.close()

    except Exception as e:
        print(f"Error occurred: {e}")


# 示例调用
def change_ma_json():
    # 定义 JSON 文件路径
    print(globalValue.BASE_HOME)
    ce_json_path = f'{globalValue.BASE_HOME}/csdb_tune/csdb_buffer_tune/ma_se/ma_ce_config.json'  # 远程 JSON 文件路径
    se_json_path = f'{globalValue.BASE_HOME}/csdb_tune/csdb_buffer_tune/ma_se/ma_se_config.json'  # 远程 JSON 文件路径

    # 调用函数更新 JSON 文件
    print(ce_json_path)
    for idx,ip in enumerate(globalValue.CONNECT_CE_IP):
        print(f'modify ce {ip} json')
        update_ce_remote_json_file(ip, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD, ce_json_path,idx)
     
    for idx,ip in enumerate(globalValue.CONNECT_SE_IP):
        print(f'modify se {ip} json')
        update_se_remote_json_file(ip, globalValue.SSH_USERNAME, globalValue.SSH_PASSWD, se_json_path,idx)