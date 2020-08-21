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
    client_list = list()
    tick_len = 1
    ticks_per_observation = 4
    memcache = list()
    memcache_last_rowid = 0

    def __init__(self, opt: dict, conf: dict):
        # Parsing options
        if 'tick_len' in opt:
            self.tick_len = opt['tick_len']
        # if 'nodeid_map' in opt:
        #     self.nodeid_map = opt['nodeid_map']
        #     self.num_ma = len(opt['nodeid_map'])
        # else:
        #     self.num_ma = opt['num_ma']
        self.tick_data_size = opt['tick_data_size']
        if 'ticks_per_observation' in opt:
            self.ticks_per_observation = opt['ticks_per_observation']
        self.observation_size = self.tick_data_size * self.ticks_per_observation

        if 'client_id' in conf['node']:
            self.client_list = list(conf['node']['client_id'].values())
            self.num_ma = len(self.client_list)
        else:
            self.client_list = None
            self.num_ma = 0

        # By default we tolerate 20% missing data tops
        self.missing_entry_tolerance = opt.get('missing_entry_tolerance',
                                               int(self.num_ma * self.ticks_per_observation * 0.2))

        # Create TABLE if not exist
        self.opt = opt
        self.conf = conf
        self.connect_db()
        c = self.conn.cursor()
        # Enable WAL mode for better concurrent read/write
        c.execute('PRAGMA journal_mode=WAL;')
        # create perfs table
        c.execute('''CREATE TABLE IF NOT EXISTS "perfs" (
                    "ts"	INTEGER,
                    "clientid"	INTEGER,
                    "pis"	BLOB,
                    PRIMARY KEY("ts","clientid"))''')
        c.execute('CREATE INDEX IF NOT EXISTS perfs_ts_index ON perfs (ts)')
        # create actions table
        c.execute('''CREATE TABLE IF NOT EXISTS "actions" (
                    "ts"	INTEGER CHECK(TYPEOF(ts)='integer'),
                    "action"	BLOB,
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

        # assert isinstance(action, dict)
        # print(f'Trying to store {action} at {ts}')
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

    def get_pi(self, node_id: int, ts: int) -> []:
        c = self.conn.cursor()
        c.execute('SELECT * FROM perfs WHERE ts = ?', (int(ts)))
        data = c.fetchone()
        data = pickle.loads(data)
        if not data:
            raise ValueError
        return data[2]

    def get_action(self, ts: int) -> int:
        c = self.conn.cursor()
        c.execute('SELECT action FROM actions WHERE ts = ?', [ts])
        action = c.fetchone()
        # Return as tuple so, require to get only first index
        # print(f'Get action: {action}')
        action = pickle.loads(action[0])
        if not action:
            return 0     # 0 is no action
        return action

    def get_action_row_count(self):
        c = self.conn.cursor()
        c.execute('SELECT COUNT(action) FROM actions')
        data = c.fetchone()
        if not data:
            raise ValueError
        return data[0]

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
        # cache size
        preloading_cache_size = len(self.memcache)
        # Getting all data from perf and actions
        c.execute('''SELECT perfs.rowid, perfs.clientid, perfs.ts, perfs.pis, action
                FROM perfs LEFT JOIN actions ON perfs.ts=actions.ts
                WHERE perfs.rowid > ? ORDER BY perfs.ts, perfs.clientid''',
                (self.memcache_last_rowid,))
        f = c.fetchall()
        # For each row
        previous = [0,0,0,0]
        for row in f:
            #set last row
            if(row[4] == None):
                continue
            self.memcache_last_rowid = max(self.memcache_last_rowid, row[0])
            # add id and pis from query data
            clientid, ts, pi_data = row[1], row[2], pickle.loads(row[3])
            # also for action
            action_data = pickle.loads(row[4]) if row[4] != None else None
            action = 0
            if(previous == [0,0,0,0]):
                action = 0
            else:
                for i,act in enumerate(zip(action_data, previous)):
                    if(act[0] != act[1]):
                        if(act[0] == None):
                            action = -1
                        action = i
                        break
            
            previous = action_data
            # check client id
            if clientid not in self.client_list:
                continue
            # check memcache id empty or memcache is not lastest ts
            if len(self.memcache) == 0 or self.memcache[-1][0] != ts:
                # memcache would have [(ts, {action}, [{pi}] * client_list)]
                self.memcache.append((ts, action, [None] * len(self.client_list)))
            self.memcache[-1][2][self.client_list.index(clientid)] = np.array(pi_data)

        # Peak memory usage (bytes on OS X, kilobytes on Linux)
        # https://stackoverflow.com/a/7669482
        logger.info('Finished loading {len} entries. Peak memory usage {size:,}.'.format(
            len=len(self.memcache) - preloading_cache_size,
            size=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        self.conn.close()

    # def get_last_time(self) -> int:
    #     """Return the last ts that has all PIs received

    #     :return: the last ts
    #     """
    #     c = self.conn.cursor()

    #     # Getting the biggest ts of the database
    #     c.execute('SELECT MIN(time) as mintime, MAX(time) as maxtime from perfs')
    #     min_time, max_time = c.fetchone()

    #     while True:
    #         # time to check if all the monitoring agents sent data at that time stamp
    #         c.execute('SELECT COUNT(node_id) from perfs WHERE time = ?', (int(max_time),))
    #         # Change it to number of all node 
    #         if self.num_node == c.fetchone()[0]:
    #             return max_ts
    #         else:
    #             if max_ts == min_ts:
    #                 raise NotEnoughDataError
    #             max_ts -= 1

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
        # TODO: Understand this function
        c = self.conn.cursor()
        # TODO: Why +10
        c.arraysize = self.num_ma * self.ticks_per_observation + 10
        
        result = np.zeros((len(self.client_list),
                        self.ticks_per_observation,
                        int(self.tick_data_size / len(self.client_list))), dtype=float)

        # Get performance indicators within time step
        # sql = '''SELECT clientid, ts, pis from perfs WHERE ts <= ? AND ts > ? ORDER BY clientid ASC, ts ASC'''
        c.execute('''SELECT clientid, ts, pis from perfs WHERE ts <= ? AND 
                ts > ? ORDER BY clientid ASC, ts ASC''',
                (ts,ts - self.ticks_per_observation))
        data = c.fetchall()
        if len(data) < self.num_ma * self.ticks_per_observation - self.missing_entry_tolerance:
            raise NotEnoughDataError
        elif len(data) != self.num_ma * self.ticks_per_observation:
            logger.debug('Observation at ts {0} has {1} missing entries'.format(
                ts, self.num_ma * self.ticks_per_observation - len(data)))
        for row in data:
            client_id = row[0]
            pi = row[2]
            if client_id not in self.client_list:
                # non client MA should send in zero length data for now
                assert len(pickle.loads(pi)) == 0
                continue
            ma_id_idx = self.client_list.index(client_id)
            ts_idx = row[1] - (ts - self.ticks_per_observation) - 1
            # numpy assignments also check that pi is in right shape
            result[ma_id_idx, ts_idx] = pickle.loads(pi)

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

    # def get_pi_ts_range(self) -> Tuple[int, int]:
    #     """Get the range of ts that has pi

    #     NotEnoughDataError will be raised if there's not enough data.

    #     :return: mints, maxts
    #     """
    #     c = self.conn.cursor()
    #     sql = 'SELECT MIN(ts) AS mints, MAX(ts) AS maxts FROM pis'
    #     c.execute(sql)
    #     data = c.fetchone()
    #     if (not data[0]) or (not data[1]):
    #         raise NotEnoughDataError('Not enough data')
    #     return data[0], data[1]

    # def get_action_ts_range(self) -> Tuple[int, int]:
    #     """Get the range of ts that has actions

    #     NotEnoughDataError will be raised if there's not enough data.

    #     :return: mints, maxts
    #     """
    #     c = self.conn.cursor()
    #     sql = 'SELECT MIN(ts) AS mints, MAX(ts) AS maxts FROM actions'
    #     c.execute(sql)
    #     data = c.fetchone()
    #     if (not data[0]) or (not data[1]):
    #         raise NotEnoughDataError('Not enough data')
    #     return data[0], data[1]
