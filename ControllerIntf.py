#!/usr/bin/env python

'''Controller Interface Daemon'''

# Import some necessary modules
import time as time
import pickle
import copy

from ceph_agent.ApiRequest import *
from ReplayDB import *
from fioClient import *

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

    # Service addr and port
    serv_addr = ''
    port = ''

    def __init__(self, conf: dict, opt: dict):
        '''
        Constructor for controller daemon

        :param opt: configuration for tuning system
        '''
        self.opt = copy.deepcopy(opt)
        self.conf = copy.deepcopy(conf)
        if('node_id' in conf.keys()):
            self.nodeid_map = conf['node_id']
    
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

        #============= Action =================
        # get dashboard address and port
        addr = self.opt['dashboard_addr']
        port = self.opt['dashboard_port']

        # Checking for certificate file for dashboard API
        cert = False
        if('use_cert' in self.opt.keys()):
            cert = self.opt['cert_file'] if self.opt['use_cert'] else False

        # Ceph implementation is here
        a = ApiRequest(addr,port, cert)

        #======================================

        # Should have a function on its own
        # Checking API
        allPath = a.paths()
        if(allPath == None):
            raise Exception('Cannot reached Dashboard API')
            return
        #========================================

        #=============== PIs ====================

        # get all client address from config file
        client_addr = [self.conf['node']['node_id'][key] for key in self.conf['node']['client_node']]

        # Create FioClient for getting PIs
        fio = FioClient(client_addr, self.opt['job_file_loc'])
        #========================================

        # Do Authentication for Dashboard
        a.auth(self.opt['auth']['username'], self.opt['auth']['passwd'])

        # setup heartbeat
        print(f"Options: {self.opt}")
        self.heartbeat = time.time()
        self.param_log = []
        self.log_times = []
        while(True):
            # flush log


            # run fio test to get PIs
            output = fio.runTest()

            # store PIs in DB
            db.insert_perf(1, int(time.time()), output)
            

            # Getting data from storage and client
            # Check data and store data in Replay DB
            # get Parameter, performance indicator and store it into ReplayDB
            log_time = time.time()
            if(int(log_time) not in self.log_times):
                self.log_times.append(int(log_time))
            for param in self.conf['ceph-param']:
                self.param_log.append(a.clusterConfig(param=param))
            db.insert_action(int(log_time), self.param_log)
                # self.param_log.append(a.clusterConfig(param=param))
                # Extract Name, type, value from each param 
            # a.performance(section='osd', '0')

            # Check health for tuning system, storage and client
            # self._health_check()

            # check time with heartbeat if less than 0.9 seconds
            if(time.time() - self.heartbeat >= 5):
                for lo_time in self.log_times:
                    print(f'At time: {lo_time} store value {db.get_action(lo_time)}')
                # for param in self.param_log:
                #     print(f"{str(param['name'])} : {str(param['type'])}")
                print('-'*50)
                # set heartbeat
                self.heartbeat = time.time()
                self.log_times = []
                self.param_log = []


    def stop(self):
        pass