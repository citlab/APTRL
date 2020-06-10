#!/usr/bin/env python

"""ASCAR ReplayDB"""

import numpy as np
import pickle
import sqlite3
from typing import *
from ascar_logging import logger

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
    ordered_client_list = None
    tick_len = 1
    ticks_per_observation = 4

    def __init__(self, opt: dict):
        # Parsing options
        # if 'tick_len' in opt:
        #     self.tick_len = opt['tick_len']
        # if 'nodeid_map' in opt:
        #     self.nodeid_map = opt['nodeid_map']
        #     self.num_ma = len(opt['nodeid_map'])
        # else:
        #     self.num_ma = opt['num_ma']
        # self.tick_data_size = opt['tick_data_size']
        # if 'ticks_per_observation' in opt:
        #     self.ticks_per_observation = opt['ticks_per_observation']
        # self.observation_size = self.tick_data_size * self.ticks_per_observation

        # if 'clients' in opt:
        #     self.ordered_client_list = [self.nodeid_map[host] for host in opt['clients']]
        #     self.ordered_client_list.sort()
        # elif self.nodeid_map:
        #     self.ordered_client_list = list(self.nodeid_map.values())
        #     self.ordered_client_list.sort()
        # else:
        #     self.ordered_client_list = None

        # # By default we tolerate 20% missing data tops
        # self.missing_entry_tolerance = opt.get('missing_entry_tolerance',
        #                                        int(self.num_ma * self.ticks_per_observation * 0.2))

        # Create TABLE if not exist
        self.connect_db(opt)
        c = self.conn.cursor()
        # Enable WAL mode for better concurrent read/write
        c.execute('PRAGMA journal_mode=WAL;')
        # performance indicators
        c.execute('''CREATE TABLE IF NOT EXISTS perfs (
                        node_id INTEGER CHECK (TYPEOF(node_id) = 'integer'),
                        time    INTEGER CHECK (TYPEOF(ts) = 'integer'),
                        perf_counter BLOB,
                        PRIMARY KEY (node_id, time))''')
        c.execute('CREATE INDEX IF NOT EXISTS perfs_time_index ON perfs (time)')
        c.execute('CREATE INDEX IF NOT EXISTS perfs_time_node_id_index ON pis (time, node_id)')
        # actions
        c.execute('''CREATE TABLE IF NOT EXISTS actions (
                        time     INTEGER PRIMARY KEY CHECK (TYPEOF(time) = 'integer'),
                        action INTEGER             CHECK (TYPEOF(action) = 'integer'))''')
        self.conn.commit()
        c.execute('ANALYZE')
        self.conn.commit()
        # The results of an ANALYZE command are only available to database connections that
        # are opened after the ANALYZE command completes.
        # https://www.sqlite.org/optoverview.html#multi_index
        self.conn.close()
        del self.conn
        self.connect_db(opt)

    def connect_db(self, opt):
        dbfile = opt['dbfile']
        # For supporting being created within DQLDaemon
        if 'disable_same_thread_check' in opt and opt['disable_same_thread_check']:
            self.conn = sqlite3.connect(dbfile, timeout=120, check_same_thread=False,
                                        detect_types=sqlite3.PARSE_COLNAMES | sqlite3.PARSE_DECLTYPES)
        else:
            self.conn = sqlite3.connect(dbfile, timeout=120,
                                        detect_types=sqlite3.PARSE_COLNAMES | sqlite3.PARSE_DECLTYPES)
        logger.info('Connected to database %s' % dbfile)

    def insert_perf(self, node_id: int, time: int, data):
        c = self.conn.cursor()
        c.arraysize = 2

        # If there's a missing entry before time, insert time as time-1
        # Checking table before insert
        c.execute('SELECT time FROM perfs WHERE node_id=? AND time>=? AND time<? ORDER BY time', (node_id, time-2, time))
        prev_t = c.fetchall()
        if len(prev_t) == 1 and prev_t[0][0] == ts-2:
            t -= 1
            logger.debug(f"A previous missing entry detected, storing PI for node_id \
                {str(node_id)} at {str(ts-1)} {str(ts)}")
        else:
            logger.debug(f"Storing PI for node_id {str(node_id)}, time {str(ts)}")

        # Insert to table
        try:
            c.execute('INSERT INTO perfs VALUES (?,?,?)', (node_id, time, data))
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            logger.warning('{type}: {msg}'.format(type=type(e).__name__, msg=str(e)))
            if 'constraint failed' in str(e):
                raise

    def insert_action(self, time: int, action: int):
        assert isinstance(action, int)
        try:
            c = self.conn.cursor()
            c.execute('INSERT INTO actions VALUES (?,?)', (ts, action))
            logger.debug(f'Stored action {action} at {time}')
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            logger.warning('{type}: {msg}'.format(type=type(e).__name__, msg=str(e)))
            if 'constraint failed' in str(e):
                raise

    def get_pi(self, node_id: int, time: float) -> []:
        c = self.conn.cursor()
        c.execute('SELECT * FROM perfs WHERE node_id = ? AND time = ?', (node_id, int(time)))
        data = c.fetchone()
        if not data:
            raise ValueError
        return data[2]

    def get_action(self, time: int) -> int:
        c = self.conn.cursor()
        c.execute('SELECT * FROM actions WHERE time = ?', [time])
        data = c.fetchone()
        if not data:
            return 0     # 0 is no action
        return data[1]

    def get_action_row_count(self):
        c = self.conn.cursor()
        c.execute('SELECT COUNT(action) FROM actions')
        data = c.fetchone()
        if not data:
            raise ValueError
        return data[0]

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

    # def get_observation(self, ts: int) -> np.ndarray:
    #     """Retrieve the PIs and CPVs for all MAs and return an observation

    #     We retrieve self.ticks_per_observe entries of data before ts from the DB
    #     for each MA, concatenate them into a list and return

    #     :param ts: the timestamp
    #     :param missing_entry_tolerance: how many missing entries are tolerated
    #     :return:
    #     """
    #     c = self.conn.cursor()
    #     c.arraysize = self.num_ma * self.ticks_per_observation + 10
    #     result = np.zeros((len(self.ordered_client_list),
    #                        self.ticks_per_observation,
    #                        int(self.tick_data_size / len(self.ordered_client_list))), dtype=float)

    #     sql = 'SELECT ma_id, ts, pi_data from pis WHERE ts <= ? AND ts > ? ORDER BY ma_id ASC, ts ASC'
    #     c.execute(sql, (ts, ts - self.ticks_per_observation))
    #     data = c.fetchall()
    #     if len(data) < self.num_ma * self.ticks_per_observation - self.missing_entry_tolerance:
    #         raise NotEnoughDataError
    #     elif len(data) != self.num_ma * self.ticks_per_observation:
    #         logger.debug('Observation at ts {0} has {1} missing entries'.format(
    #             ts, self.num_ma * self.ticks_per_observation - len(data)))
    #     for row in data:
    #         ma_id = row[0]
    #         pi = row[2]
    #         if ma_id not in self.ordered_client_list:
    #             # non client MA should send in zero length data for now
    #             assert len(pickle.loads(pi)) == 0
    #             continue
    #         ma_id_idx = self.ordered_client_list.index(ma_id)
    #         ts_idx = row[1] - (ts - self.ticks_per_observation) - 1
    #         # numpy assignments also check that pi is in right shape
    #         result[ma_id_idx, ts_idx] = pickle.loads(pi)

    #     return result.reshape((self.observation_size,))

    # def get_last_n_observation(self, n: int=1) -> List[np.ndarray]:
    #     result = []
    #     c = self.conn.cursor()
    #     c.execute('SELECT MIN(ts) as mints, MAX(ts) as maxts from pis')
    #     min_ts, max_ts = c.fetchone()
    #     while True:
    #         try:
    #             result.append(self.get_observation(max_ts))
    #             if len(result) == n:
    #                 return result
    #         except NotEnoughDataError:
    #             if max_ts == min_ts:
    #                 raise
    #         max_ts -= 1

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
