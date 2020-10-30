#!/usr/bin/env python

'''Controller Interface Daemon'''

# Import some necessary modules
from time import sleep
from ascar_logging import flush_log
import time as time
import pickle
import copy
import requests

import socket
import paramiko

from ceph_agent.ApiRequest import *
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
        if('node_id' in conf.keys()):
            self.nodeid_map = conf['node_id']
            
        # get dashboard address and port
        addr = self.opt['dashboard_addr']
        port = self.opt['dashboard_port']

        # Checking for certificate file for dashboard API
        cert = False
        if('use_cert' in self.opt.keys()):
            cert = self.opt['cert_file'] if self.opt['use_cert'] else False

        # Ceph implementation is here
        self.api = ApiRequest(addr,port, cert)
        
        # Checking API
        allPath = self.api.paths()
        if(allPath == None):
            raise Exception('Cannot reached Dashboard API')
        
        # Do Authentication for Dashboard
        self.api.auth(self.opt['auth']['username'], self.opt['auth']['passwd'])
    
    def _health_check(self) -> str:
        '''
        Check health for controller interface and each node status
        '''
        if not self.nodeid_map:
            result = 'nodeid_map is missing. '
            return result

        else:
            for node,status in self.node_status.items():
                result += f'{node}: {status}'
            return result
    
    def _handle_node_status(self):
        '''
        Handle status from each node by asking the storage system
        ''' 
        # get health status from Storage system
        # try to make it general for any kind of storage system
        pass

    def start(self):
        '''
        Start the interface daemon and start asking storage system
        '''
        # assert something here! Check if storage is reachable

        # Start DBReplay here
        # ReplayDB must be created in start(), which may be run in a separate thread.
        # SQLite doesn't like the DBConn be created in different threads.
        db = ReplayDB(self.opt, self.conf)
        db.connect_db()

        try:
            # setup heartbeat
            print(f"Options: {self.opt}")
            # self.heartbeat = time.time()
            self.param_log = []
            self.log_times = []
        except requests.exceptions.ConnectionError as e:
            logger.error(f'Error exception: {e}')
        
        # get all client address from config file
        client_addr = list(self.conf['node']['client_addr'].values())
        client_node = list(self.conf['node']['client_addr'].keys())
        
        while(True):
            # flush log
            flush_log()
            log_time = int(time.time())
            for i,client in enumerate(client_addr):
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
                    
                    if(int(log_time) not in self.log_times):
                        self.log_times.append(log_time)
                    # store PIs in DB (node_id, time, data)
                    node_name = client_node[i]
                    db.insert_perf(self.conf['node']['client_id'][node_name], log_time, pi_data)
                    # Close connection with client
                    conn.close()
                    # sleep(self.opt['heartbeat'])
                except ConnectionRefusedError:
                    logger.info(f'address {addr} is not available')
            sleep(self.opt['heartbeat'])
            # TODO: Read this and implement necessary line

            # Check health for tuning system, storage and client
            self._health_check()

            # check time with heartbeat if less than 0.9 seconds
            # if(time.time() - self.heartbeat >= 5):
            #     for lo_time in self.log_times:
            #         print(f'At time: {lo_time} store value {db.get_action(lo_time)}')

            #     print('-'*50)
            #     # set heartbeat
            #     self.heartbeat = time.time()
            #     self.log_times = []
            #     self.param_log = []
                
        self.conn.setblocking(True)

    def createConnection(self, addr, port):
        conn = socket.create_connection((addr,port), 30)
        return conn
    
    @staticmethod
    def broadcastAction(action, ts, conf, opt):
        
        addr = opt['dashboard_addr']
        port = opt['dashboard_port']

        # # Checking for certificate file for dashboard API
        # cert = False
        # if('use_cert' in opt.keys()):
        #     cert = opt['cert_file'] if opt['use_cert'] else False

        # # Ceph implementation is here
        # api = ApiRequest(addr,port, cert)
        
        # # Checking API
        # allPath = api.paths()
        # if(allPath == None):
        #     raise Exception('Cannot reached Dashboard API')
        
        # # Do Authentication for Dashboard
        # api.auth(opt['auth']['username'], opt['auth']['passwd'])
        for idx, param in enumerate(conf['ceph-param']):
            
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect('10.162.230.51', username='root', password='hammer23')
            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(f'ceph config set global {list(param.keys())[0]} {str(action[idx])}')
            ssh.close()
            
            # api.clusterConfig(list(param.keys())[0], str(action[idx]), section='global')
    
    def stop(self):
        pass