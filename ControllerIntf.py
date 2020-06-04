#!/usr/bin/env python

'''Controller Interface Daemon'''

# Import some necessary modules
import time
import pickle
import copy

from ceph_agent.ApiRequest import *
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

    def __init__(self, opt: dict, conf: dict):
        '''
        Constructor for controller daemon

        :param opt: configuration for tuning system
        '''
        self.opt = copy.deepcopy(opt)
        self.conf = copy.deepcopy(conf)
        if('node_id' in conf.keys()):
            self.nodeid_map = conf['node_id']
        # if('service_addr' in conf.keys()):
        #     self.serv_addr = conf['service_addr']
        # if('port' in conf.keys()):
        #     self.port = conf['port']
        # else:
        #     self.port = 8080
    
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
        # assert something here!
        # assert

        addr = self.opt['dashboard_addr']
        port = self.opt['dashboard_port']
        cert = False

        if('use_cert' in self.opt.keys()):
            cert = self.opt['cert_file'] if self.opt['use_cert'] else False

        a = ApiRequest(addr,port, cert)
        allPath = a.paths()
        print(allPath)
        a.auth('admin', 'admin')
        # a.clusterConfig(param='osd_recovery_sleep', val='0', section='global')
        # #print(a.clusterConfig(param='osd_recovery_sleep'))

        # #print(a.health(report='minimal'))
        # print(a.performance('mon','a'))

        # Create database for storing data from storage system
        # db = ReplayDB()
        print(f"Hello world{self.opt}")
        self.heartbeat = self.opt['heartbeat']
        while(True):
            time.sleep(self.heartbeat)
            print(a.clusterConfig(param='osd_recovery_sleep'))


    def stop(self):
        pass
