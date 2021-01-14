import os
import time
import paramiko

def wait(seconds):
    for i in reversed(range(seconds)):
        print(f"Time left: {i+1} second(s)", end="\r")
        time.sleep(1)
        print("                                   ", end="\r")

def set_param_default():
    default = [4294967296,33554432,4096,0]
    params = ['osd_memory_target','rbd_cache_size',
    'ms_tcp_prefetch_max_size','ms_tcp_rcvbuf']
    try:
        for i,param in enumerate(params):
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            ssh.connect('loewe.arch.suse.de', username='root', port=2789)
            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(f'ceph config set global {param} {default[i]}')
            ssh.close()
    except Exception as e:    
        logger.info(f'Cannot connect to storage. Parameter won\'t update')
        # api.clusterConfig(list(param.keys())[0], str(action[idx]), section='global')

# Run tuning system with control interface (APTRL)
os.system("/home/pkhantiviriya/myAPTRL/run_intf.sh /home/pkhantiviriya/myAPTRL/config.yml start")
wait(30)
os.system("/home/pkhantiviriya/myAPTRL/run_ml.sh /home/pkhantiviriya/myAPTRL/config.yml start")

# wait(18000)

# # Run tuning system with control interface (CAPES)
# os.system("/home/pkhantiviriya/myAPTRL/run_intf.sh /home/pkhantiviriya/myAPTRL/config.yml stop")
# os.system("/home/pkhantiviriya/myAPTRL/run_ml.sh /home/pkhantiviriya/myAPTRL/config.yml stop")

# set_param_default()

# os.system("/home/pkhantiviriya/APTRL/run_intf.sh /home/pkhantiviriya/APTRL/config.yml start")
# wait(30)
# os.system("/home/pkhantiviriya/APTRL/run_ml.sh /home/pkhantiviriya/APTRL/config.yml start")

# wait(18000)

# # Run control interface (Only Ceph)
# os.system("/home/pkhantiviriya/APTRL/run_intf.sh /home/pkhantiviriya/APTRL/config.yml stop")
# os.system("/home/pkhantiviriya/APTRL/run_ml.sh /home/pkhantiviriya/APTRL/config.yml stop")

# set_param_default()

# os.system("/home/pkhantiviriya/myAPTRL/run_intf.sh /home/pkhantiviriya/myAPTRL/config.yml start")

# wait(18000)

# os.system("/home/pkhantiviriya/myAPTRL/run_intf.sh /home/pkhantiviriya/myAPTRL/config.yml stop")

# set_param_default()
print("Process is done")