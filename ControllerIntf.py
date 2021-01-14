#!/usr/bin/env python
 
'''Controller Interface Daemon'''
 
# Import some necessary modules
from time import sleep
from ascar_logging import flush_log
import time as time
import pickle
import copy
import requests
 
# Checking the running OS
import platform
import subprocess
 
import socket
import paramiko
 
from ReplayDB import *
 
__autor__ = 'Puriwat Khantiviriya'
 
class ControllerInft:
    '''The Controller daemon
 
    a daemon for controlling the tuning system and managing 
    the communication with storage system
    '''
 
    nodeid_map = None
    opt = None
    socket = None
    started = False
    health_status = ''
    node_status = dict()
 
    def __init__(self, conf: dict, opt: dict):
        '''
        Constructor for controller daemon
 
        :param opt: configuration for tuning system
        '''
        self.opt = copy.deepcopy(opt)
        self.conf = copy.deepcopy(conf)
 
        if('node_name' in conf['node'].keys()):
            self.nodeid_map = conf['node']['node_name']
        self.node_tolerance_map = [0] * len(self.nodeid_map)
        
        # Get all available server ip and port
        self.srvs = list(self.conf['node']['server_addr'].values())
        # Get all available client ip and port
        self.clients = list(self.conf['node']['client_addr'].values())
 
    def _ping_addr(self, addr):
        '''
            Sending ping ICMP request to addr to 
            check health of the host
 
            :param addr: Host's address to check
 
            Return True if host responds to a ping request
        '''
        num = '-n' if platform.system().lower()=='windows' else '-c'
        cmd = ['ping', num, '1', addr]
        return subprocess.call(cmd) == 0
 
 
    def _health_check(self) -> str:
        '''
        Check health for controller interface and each node status
        '''
        # Loop each node to check health by ping
        for i,node in enumerate(self.nodeid_map):
            if(node in self.conf['node']['server_addr'].keys()):
                node_addr = self.conf['node']['server_addr'][node]
            elif(node in self.conf['node']['client_addr'].keys()):
                node_addr = self.conf['node']['client_addr'][node]
            else:
                self.nodeid_map.remove(node)
                self.node_tolerance_map.pop(i)
 
            # Try ping the node
            addr = node_addr.split(':')
            if(self._ping_addr(addr[0])):
                pass
            else:
                if(self.node_tolerance_map[i] >= self.opt['ping_miss']):
                    self.nodeid_map.remove(node)
                    self.node_tolerance_map.pop(i)
                else:
                    self.node_tolerance_map[i] += 1
 
    def start(self):
        '''
        Start the interface daemon and start asking storage system
        '''
 
        # Start DBReplay here
        # ReplayDB must be created in start(), which may be run in a separate thread.
        # SQLite doesn't like the DBConn be created in different threads.
        db = ReplayDB(self.opt, self.conf)
        db.connect_db()
 
        # get all client address from config file
        client_node = list(self.conf['node']['client_addr'].keys())
        
        while(True):
            # flush log
            flush_log()
            log_time = int(time.time())
            for client in self.clients:
                # split address and port
                addr,port = client.split(':')
 
                try:
                    # create connection with each client
                    conn = self.createConnection(addr,port)
                    logger.info(f'Connected to {addr} via port {port}')
                    
                    # requesting for lastest PI
                    conn.sendall(b'REQ_PI')
                    pi_data = pickle.loads(conn.recv(1024))
                    logger.info(f'Received {pi_data}')
                    
                    # store PIs in DB (node_id, time, data)
                    db.insert_perf(self.conf['node']['client_id'][client], log_time, pi_data)
                    # Close connection with client
                    conn.close()
 
                except ConnectionRefusedError:
                    logger.info(f'address {addr} is not available')
                except ConnectionResetError:
                    logger.info(f'address {addr} is reset by peer')
                except EOFError:
                    logger.info(f'address {addr} send too much input')
                except socket.timeout:
                    logger.info(f'address {addr} not responding')
                except socket.gaierror:
                    logger.info(f'address {addr} not known or cannnot be access')
 
            # Check health of each node
            # self._health_check()
            if(time.time() - self.opt['heartbeat'] < log_time):
                sleep(self.opt['heartbeat'])
            
        self.conn.setblocking(True)
 
    def createConnection(self, addr, port):
        conn = socket.create_connection((addr,port), 30)
        return conn
    
    @staticmethod
    def broadcastAction(action, ts, conf, opt):
        
        srv = conf['node']['server_addr']['master'].split(":")
        addr = srv[0]
        port = srv[1]
 
        try:
            for idx, param in enumerate(conf['ceph-param']):
                logger.info(f'Update {list(param.keys())[0]} with value {str(action[idx])}')
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                
                ssh.connect(addr, username=opt['srv_usr'], port=port)
                ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(f'ceph config set global {list(param.keys())[0]} {str(action[idx])}')
                ssh.close()
        except Exception as e:    
            logger.info(f'Cannot connect to storage. Parameter won\'t update')
            # api.clusterConfig(list(param.keys())[0], str(action[idx]), section='global')
    
    def stop(self):
        pass

