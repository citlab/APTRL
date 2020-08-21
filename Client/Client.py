#!/usr/bin/env python

'''Client'''

# Import some necessary modules
import socket
import subprocess
import json
import pickle

__autor__ = 'Puriwat Khantiviriya'

class Client:
    
    # Service addr and port
    serv_addr = '127.0.0.1'
    port = 7658
    
    # command for running fio on client side
    cmd = ['/usr/local/bin/fio', 
        '--filename=/home/pkhantiviriya/rbd0/deleteme',
        '--direct=1', '--rw=write', '--bs=4k', 
        '--size=500M', '--iodepth=16', '--name=write4k',
        '--output-format=terse', '--status-interval=1']
    
    out_col = ['terse_version_3', 'fio_version', 'jobname', 'groupid', 'error',
        'read_kb', 'read_bandwidth', 'read_iops', 'read_runtime_ms',
        'read_slat_min', 'read_slat_max', 'read_slat_mean', 'read_slat_dev', 
        'read_clat_min', 'read_clat_max', 'read_clat_mean', 'read_clat_dev', 
        'read_clat_pct01', 'read_clat_pct02', 'read_clat_pct03', 
        'read_clat_pct04', 'read_clat_pct05', 'read_clat_pct06', 
        'read_clat_pct07', 'read_clat_pct08', 'read_clat_pct09', 
        'read_clat_pct10', 'read_clat_pct11', 'read_clat_pct12', 
        'read_clat_pct13', 'read_clat_pct14', 'read_clat_pct15', 
        'read_clat_pct16', 'read_clat_pct17', 'read_clat_pct18', 
        'read_clat_pct19', 'read_clat_pct20', 'read_tlat_min', 
        'read_lat_max', 'read_lat_mean', 'read_lat_dev', 'read_bw_min', 
        'read_bw_max', 'read_bw_agg_pct', 'read_bw_mean', 'read_bw_dev', 
        'write_kb', 'write_bandwidth', 'write_iops', 'write_runtime_ms', 
        'write_slat_min', 'write_slat_max', 'write_slat_mean', 
        'write_slat_dev', 'write_clat_min', 'write_clat_max', 
        'write_clat_mean', 'write_clat_dev', 'write_clat_pct01', 
        'write_clat_pct02', 'write_clat_pct03', 'write_clat_pct04', 
        'write_clat_pct05', 'write_clat_pct06', 'write_clat_pct07', 
        'write_clat_pct08', 'write_clat_pct09', 'write_clat_pct10', 
        'write_clat_pct11', 'write_clat_pct12', 'write_clat_pct13', 
        'write_clat_pct14', 'write_clat_pct15', 'write_clat_pct16', 
        'write_clat_pct17', 'write_clat_pct18', 'write_clat_pct19', 
        'write_clat_pct20', 'write_tlat_min', 'write_lat_max', 
        'write_lat_mean', 'write_lat_dev', 'write_bw_min', 'write_bw_max', 
        'write_bw_agg_pct', 'write_bw_mean', 'write_bw_dev', 'cpu_user', 
        'cpu_sys', 'cpu_csw', 'cpu_mjf', 'cpu_minf', 'iodepth_1', 
        'iodepth_2', 'iodepth_4', 'iodepth_8', 'iodepth_16', 'iodepth_32', 
        'iodepth_64', 'lat_2us', 'lat_4us', 'lat_10us', 'lat_20us', 
        'lat_50us', 'lat_100us', 'lat_250us', 'lat_500us', 'lat_750us', 
        'lat_1000us', 'lat_2ms', 'lat_4ms', 'lat_10ms', 'lat_20ms', 
        'lat_50ms', 'lat_100ms', 'lat_250ms', 'lat_500ms', 'lat_750ms', 
        'lat_1000ms', 'lat_2000ms', 'lat_over_2000ms', 'disk_name', 
        'disk_read_iops', 'disk_write_iops', 'disk_read_merges', 
        'disk_write_merges', 'disk_read_ticks', 'write_ticks', 
        'disk_queue_time', 'disk_util']

    def __init__(self):
        
        pi_arr = []
        
        # self.soc = self.connectSocket(self.serv_addr, self.port)
        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        soc.bind((self.serv_addr, self.port))

        soc.listen(0)
        # conn, addr = soc.accept()
        # print(f'Connected with {addr}')
        
        proc = self.executeProcess()
        for stdou in iter(proc.stdout.readline, ""):
            try:
                # if(conn.fileno()):
                conn, addr = soc.accept()
                print(f'Connected with {addr}')
                # try:
                data = conn.recv(1024)
                print(data)
                if(data == b'REQ_PI'):
                    pi_arr = self.getPI(stdou, proc)
                    print(pi_arr)
                    conn.sendall(pickle.dumps(pi_arr))
                conn.close()
            except Exception:
                print('data not received')
                
            # except EOFError:
            #     conn,addr = soc.accept()
        # if self.timeout > 0 and time.time() - start_time >= self.timeout:
        #     soc.close()
        #     raise RuntimeError("Timeout exceeded (%ds)" % self.timeout)
            
        
        
        # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        #     s.connect((HOST, PORT))
        #     s.sendall(b'Hello, world')
        #     data = s.recv(1024)

        # print('Received', repr(data))
        
    def executeProcess(self):
        return subprocess.Popen(self.cmd, stdout=subprocess.PIPE, universal_newlines=True)
    
    def getPI(self, line, proc):
        # if(proc.poll() == None):
        pi_data = line.split(';')
        pi_arr = []
        
        pi_arr.append(pi_data[self.out_col.index('read_kb')])
        pi_arr.append(pi_data[self.out_col.index('write_kb')])
        pi_arr.append(pi_data[self.out_col.index('read_lat_mean')])
        pi_arr.append(pi_data[self.out_col.index('write_lat_mean')])
            
        return pi_arr
        # else:
        #     return None
            # raise Exception("Process is finished")          

        
    # def connectSocket(self, serv_addr, port):
    #     # logger.debug("waiting to connect with client")
        
    #     # soc = socket.create_connection((serv_addr,port), 30)
    #     # soc.connect((serv_addr, port))
    #     return soc
        
    def send(self,pi_data):
        self.soc.sendall(pickle.dumps(pi_data))
        data = self.soc.recv(1024)
        print('Received', repr(data))
        
Client()