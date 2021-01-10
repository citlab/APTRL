#!/usr/bin/env python

'''Client'''

# Import some necessary modules
import asyncio          #Running asynchronized socket
import nest_asyncio
nest_asyncio.apply()

import socket
import subprocess           # Running command
import json
import pickle           # Covert or dump binary
import datetime         # Time stamp
import sys, getopt          # Accepting arguments in cli

__autor__ = 'Puriwat Khantiviriya'

class Client:
    
    # request queue for socket connection
    req_queue = []
    # Service addr and port
    serv_addr = socket.gethostname()
    port = 7658

    # command for running fio on client side
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

    def __init__(self, filesize):
        self.pi_arr = []
        self.prev_pi = []
        self.filesize = filesize
        self.srv = None
        self.loop = None
        self.result = None
        self.proc = self.execute_process(filesize)
        self.proc_iter = iter(self.proc.stdout.readline, "")

    def set_srv(self, loop, srv, coro):
        self.srv = srv
        self.loop = loop
        self.coro = coro

    @asyncio.coroutine
    def handle_req(self, reader, writer):
        self.req_queue.append((reader, writer))
        print(f"Connected with {writer.get_extra_info('peername')}")
        # self.checkQueue(self.proc_iter)
        try:
            out_line = next(self.proc_iter)
            self.pi_arr = self.get_pi(out_line)
            if(self.pi_arr == None):
                    self.pi_arr = self.prev_pi
            self.store_perf()
            self.prev_pi = self.pi_arr

            # Get connection from queue and check the message
            reader, writer = self.req_queue.pop(0)
            data = yield from reader.read(1024)
            msg = data.decode()
            print(msg)
            if(msg == 'REQ_PI'):
                # retrieve PI from cmd
                print(f"Sending: {self.pi_arr}")
                writer.write(pickle.dumps(self.pi_arr[1:]))
                yield from writer.drain()

                writer.close()
            else:
                # Do nothing
                writer.close()
        except StopIteration:
            print(f'process is done.')
            self.coro.close()
            self.srv.close()
            self.loop.run_until_complete(srv.wait_closed())
        except Exception as e:
            print(f'Error occured: {e}')

    def execute_process(self, size):
        cmd = ['fio','/root/client/syn_workload_profile.fio',
                '--filename=/root/client/mountfs/deleteme3',f'--size={size}',
                '--output-format=terse', '--status-interval=1']
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    
    def get_pi(self, line):
        pi_data = line.split(';')
        if(len(pi_data) >= 121):
            pi_arr = []

            pi_arr.append(pi_data[self.out_col.index('jobname')])
        
            pi_arr.append(pi_data[self.out_col.index('read_bandwidth')])
            pi_arr.append(pi_data[self.out_col.index('write_bandwidth')])
            pi_arr.append(pi_data[self.out_col.index('read_lat_mean')])
            pi_arr.append(pi_data[self.out_col.index('write_lat_mean')])
            
            return pi_arr
        else:
            return None
        
    def store_perf(self):
        print(len(self.pi_arr))
        if(len(self.pi_arr) != 0):
            filename=self.pi_arr[0]+f'_{self.filesize}'+'.txt'
            now = datetime.datetime.now()
            with open(filename, 'a') as outf:
                outf.write(f'{now},{self.pi_arr[1]},{self.pi_arr[2]},{self.pi_arr[3]},{self.pi_arr[4]}\n')


size = 0
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "hs:", ["size="])
except getopt.GetoptError:
    print('''
        Client.py -s <filesize>, ... [K,M,G,T,P,Ki,Mi,Gi,Ti,Pi]
        Example:
            Client.py -s 1G
            Client.py -s 1G,2M,3K
    ''')
    sys.exit(2)

for opt, arg in opts:
    if opt in ("-h", "--help"):
        print('''
            Client.py -s <filesize>, ... [K,M,G,T,P,Ki,Mi,Gi,Ti,Pi]
            Example:
                Client.py -s 1G
                Client.py -s 1G,2M,3K
        ''')
    elif opt in ("-s", "--size"):
        sizes = arg.split(',')
        for size in sizes:

            client = Client(size)

            loop = asyncio.get_event_loop()
            coro = asyncio.start_server(client.handle_req, client.serv_addr, 
                client.port, loop=loop)
            srv = loop.run_until_complete(coro)
            client.set_srv(loop, srv, coro)
            print(f'Open connection on {srv.sockets[0].getsockname()}')
            try:
                loop.run_forever()
            except KeyboardInterrupt:
                pass
                
            srv.close()
            loop.run_until_complete(srv.wait_closed())
            loop.close()

    
