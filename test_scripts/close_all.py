from maEnv import utils
import globalValue


IP = [
    "172.81.0.10",
    "172.81.0.11",
    "172.81.0.12",
    "172.81.0.13",
    "172.81.0.14",
    "172.81.0.15",
    "172.81.0.16",
    "172.81.0.17",
    "172.81.0.10",
    "172.81.0.19",
    "172.81.0.20",
    "172.81.0.21",
]

PORT = [
    2000,
    2000,
    2000,
    2000,
    2000,
    2000,
    2000,
    2000,
    2000,
    2000,
    4000,
    4000,
]



for i in range(12):
    utils.send_msg_to_server("exit", IP[i], PORT[i])
    utils.sshExe(IP[i], globalValue.SSH_USERNAME, globalValue.SSH_PASSWD,
                    globalValue.MYSQLD_CE_CLOSE_EXEC)
        