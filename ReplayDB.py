#!/usr/bin/env python
 
"""ASCAR ReplayDB"""
 
import numpy as np
import pickle
import sqlite3
from sqlite3 import Error
from typing import *
from ascar_logging import logger
import resource
 
__author__ = 'Yan Li'
__copyright__ = 'Copyright (c) 2016, 2017 The Regents of the University of California. All rights reserved.'
 
 
class NotEnoughDataError(BaseException):
    def __init__(self, *args, **kwargs):
        BaseException.__init__(self, *args, **kwargs)
 
 
class ReplayDB:
    """A class for accessing ReplayDB
 
    Attributes:
        ordered_client_list: a sorted list of all MA IDs. So far because only client MA sends in data, we store only
                            client IDs.
 
    :type conn: sqlite3.Connection
    :type nodeid_map: Dict[str, int]
    :type ordered_client_list: List[int]
    """
    conn = None
    nodeid_map = None
    ticks_per_observation = 1
    client_list = []
    memcache = []
    memcache_last_rowid = 0
 
    def __init__(self, opt: dict, conf: dict):
        # Parsing list of client_id and number of all clients
        if 'client_id' in conf['node']:
            self.client_list = list(conf['node']['client_id'].values())
            self.num_clients = len(self.client_list)
        else:
            self.client_list = None
            self.num_clients = 0
 
        self.ticks_per_observation = opt['ticks_per_observation']
 
        # Parsing observation size
        self.observation_size = self.num_clients * len(conf['ceph-param']) * self.ticks_per_observation
 
        # By default we tolerate 20% missing data tops
        self.missing_entry_tolerance = opt.get('missing_entry_tolerance',
                                               int(self.num_clients * 0.2))
 
        # loading opt and conf in config file
        self.opt = opt
        self.conf = conf
 
        # Create table on the database
        self.connect_db()
        c = self.conn.cursor()
        # Enable WAL mode for better concurrent read/write
        c.execute('PRAGMA journal_mode=WAL;')
        # create perfs table
        c.execute('''CREATE TABLE IF NOT EXISTS "perfs" (
                    "ts"    INTEGER,
                    "clientid"  INTEGER,
                    "pis"   BLOB,
                    PRIMARY KEY("ts","clientid"))''')
        c.execute('CREATE INDEX IF NOT EXISTS perfs_ts_index ON perfs (ts)')
        # create actions table
        c.execute('''CREATE TABLE IF NOT EXISTS "actions" (
                    "ts"    INTEGER CHECK(TYPEOF(ts)='integer'),
                    "action"    BLOB,
                    PRIMARY KEY("ts"))''')
        self.conn.commit()
        c.execute('ANALYZE')
        self.conn.commit()
        # The results of an ANALYZE command are only available to database connections that
        # are opened after the ANALYZE command completes.
        # https://www.sqlite.org/optoverview.html#multi_index
        self.conn.close()
        del self.conn
        logger.info(f"Connected to database {self.conf['replaydb']['dbfile']}")
 
    def connect_db(self):
        dbfile = self.conf['replaydb']['dbfile']
        # For supporting being created within DQLDaemon
        if 'disable_same_thread_check' in self.opt and self.opt['disable_same_thread_check']:
            self.conn = sqlite3.connect(dbfile, timeout=120, check_same_thread=False,
                                        detect_types=sqlite3.PARSE_COLNAMES | sqlite3.PARSE_DECLTYPES)
        else:
            self.conn = sqlite3.connect(dbfile, timeout=120,
                                        detect_types=sqlite3.PARSE_COLNAMES | sqlite3.PARSE_DECLTYPES)
 
    def insert_perf(self, node_id: int, ts: int, data):
        c = self.conn.cursor()
        c.arraysize = 3
 
        # If there's a missing entry before time, insert time as time-1
        # Checking table before insert
        c.execute('SELECT ts FROM perfs WHERE ts>=? AND ts<? ORDER BY ts', (ts-2, ts))
        prev_t = c.fetchall()
        if len(prev_t) == 1 and prev_t[0][0] == ts-2:
            ts -= 1
            logger.debug(f"A previous missing entry detected, storing PI at {str(ts-1)} {str(ts)}")
        else:
            logger.debug(f"Storing PI at ts {str(ts)}")
 
        # Insert to table
        try:
            c.execute('INSERT INTO perfs VALUES (?,?,?)', (ts, node_id, pickle.dumps(data)))
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            logger.warning('{type}: {msg}'.format(type=type(e).__name__, msg=str(e)))
            if 'constraint failed' in str(e):
                raise
 
    def insert_action(self, ts: int, action):
        try:
            c = self.conn.cursor()
            action_pickle = pickle.dumps(action)
            c.execute('INSERT INTO actions VALUES (?,?)', (ts, action_pickle))
            print(f'Stored action at {ts} in DB')
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            logger.warning('{type}: {msg}'.format(type=type(e).__name__, msg=str(e)))
            if 'constraint failed' in str(e):
                raise
 
    def refresh_memcache(self):
        """Refresh memcache 
        
        Refresh memcache by moving row index and retrieve new data from replay db
        """
        
        self.connect_db()
        logger.info('Loading cache...')
        # Init cursor for database
        c = self.conn.cursor()
        # Use a large arraysize to increase read speed; we don't care about memory usage
        c.arraysize = 1000000
 
        # if memcache is not initialize yet
        if not self.memcache:
            self.memcache = list()
        else:
            for cache in self.memcache:
                # If action is empty or None
                if cache[1] == []:
                    self.memcache.remove(cache)
                elif cache[1] == [None] * len(self.conf['ceph-param']):
                    self.memcache.remove(cache)
        
        # cache size
        preloading_cache_size = len(self.memcache)
        
        # Getting all data from perf and actions
        c.execute('''SELECT perfs.rowid, perfs.clientid, perfs.ts, perfs.pis, action
                FROM perfs LEFT JOIN actions ON perfs.ts=actions.ts
                WHERE perfs.rowid > ? ORDER BY perfs.ts, perfs.clientid''',
                (self.memcache_last_rowid,))
        f = c.fetchall()
        # For each row
        previous = [0] * len(self.conf['ceph-param'])
        for row in f:
            self.memcache_last_rowid = max(self.memcache_last_rowid, row[0])                                    # Update last row id
            clientid, ts, pi_data = row[1], row[2], pickle.loads(row[3])                                        # add id and pis from query data
            
            action_data = pickle.loads(row[4]) if row[4] != None else [None] * len(self.conf['ceph-param'])     # also for action
            act_idx = []
 
            # Check which parameters are updated
            # First row
            if(previous == [0] * len(self.conf['ceph-param'])):
                pass
            # Other row
            else:
                for i,act in enumerate(zip(action_data, previous)):
                    if(act[0] != act[1]) and (act[1] != None):
                        act_idx.append(i)
            
            previous = action_data                  # Update previous action
 
            # check client id
            if clientid not in self.client_list:
                continue
            # check memcache id empty or memcache is not lastest ts
            if len(self.memcache) == 0 or self.memcache[-1][0] != ts:
                # memcache would have [(ts, {action}, [{pi}] * client_list)]
                self.memcache.append((ts, act_idx, [None] * len(self.client_list)))
            # Update pi data to memcache
            self.memcache[-1][2][self.client_list.index(clientid)] = np.array(pi_data)
 
        # Peak memory usage (bytes on OS X, kilobytes on Linux)
        # https://stackoverflow.com/a/7669482
        logger.info('Finished loading {len} entries. Peak memory usage {size:,}.'.format(
            len=len(self.memcache) - preloading_cache_size,
            size=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        self.conn.close()
 
    def get_observation(self, ts: int) -> np.ndarray:
        """Retrieve the PIs and CPVs for all MAs and return an observation
 
        We retrieve self.ticks_per_observe entries of data before ts from the DB
        for each MA, concatenate them into a list and return
 
        Args:
            ts (int): the timestamp
 
        Raises:
            NotEnoughDataError: [description]
 
        Returns:
            np.ndarray: [description]
        """
        
        self.connect_db()
        c = self.conn.cursor()
        c.arraysize = self.num_clients * self.ticks_per_observation + 10
 
        # Create np array of size [clients x ticks_per_observation x parameters]
        result = np.zeros((len(self.client_list),
                        self.ticks_per_observation,
                        len(self.conf['ceph-param'])), dtype=float)
 
        # Get performance indicators within time step
        # sql = '''SELECT clientid, ts, pis from perfs WHERE ts <= ? AND ts > ? ORDER BY clientid ASC, ts ASC'''
        c.execute('''SELECT clientid, ts, pis from perfs WHERE ts <= ? AND 
                ts > ? ORDER BY clientid ASC, ts ASC''',
                (ts,ts - self.ticks_per_observation))
 
        data = c.fetchall()
        if len(data) < self.num_clients * self.ticks_per_observation - self.missing_entry_tolerance:
            raise NotEnoughDataError
        elif len(data) != self.num_clients * self.ticks_per_observation:
            logger.debug('Observation at ts {0} has {1} missing entries'.format(
                ts, self.num_clients * self.ticks_per_observation - len(data)))
        for row in data:
            client_id = row[0]
            pi = row[2]
            if client_id not in self.client_list:
                # non client MA should send in zero length data for now
                assert len(pickle.loads(pi)) == 0
                continue
 
            client_id_idx = self.client_list.index(client_id)
            ts_idx = row[1] - (ts - self.ticks_per_observation) - 1
            # numpy assignments also check that pi is in right shape
            result[client_id_idx, ts_idx] = pickle.loads(pi)
 
        self.conn.close()
        return result.reshape((self.observation_size,))
 
    def get_last_n_observation(self, n: int=1) -> List[np.ndarray]:
        """get last n observation
 
        get observation of last n observations
 
        Args:
            n (int, optional): [description]. Defaults to 1.
 
        Returns:
            List[np.ndarray]: [description]
        """
        self.connect_db()
        result = []
        c = self.conn.cursor()
        # Get minimum ts and maximum ts from perfs table
        c.execute('SELECT MIN(ts) as mints, MAX(ts) as maxts from perfs')
        # fetch execution into min and max ts
        min_ts, max_ts = c.fetchone()
        while True:
            try:
                # append observation from max_ts
                result.append(self.get_observation(max_ts))
                # check if getting enough observation
                if len(result) == n:
                    self.conn.close()
                    return result
            except NotEnoughDataError:
                if max_ts == min_ts:
                    raise
            # keep subtract max_ts until min_ts
            max_ts -= 1

