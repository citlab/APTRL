#!/usr/bin/env python

"""ASCAR Deep Q Learning Daemon.

Copyright (c) 2016, 2017 The Regents of the University of California. All
rights reserved.

Created by Yan Li <yanli@tuneup.ai>, Kenneth Chang <kchang44@ucsc.edu>,
Oceane Bel <obel@ucsc.edu>. Storage Systems Research Center, Baskin School
of Engineering.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions bof source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Storage Systems Research Center, the
      University of California, nor the names of its contributors
      may be used to endorse or promote products derived from this
      software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
REGENTS OF THE UNIVERSITY OF CALIFORNIA BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.

Some of the code are based on https://github.com/nivwusquorum/tensorflow-deepq under
the following license:

The MIT License (MIT)

Copyright (c) 2015 Szymon Sidor

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.
"""

import gc
import os
import tempfile
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
import time
import traceback
from ascar_logging import *
from tf_rl.controller import DiscreteDeepQ
from tf_rl.models import MLP
import resource
import random
import json

from ReplayDB import *


__author__ = 'Yan Li'
__copyright__ = 'Copyright (c) 2016, 2017 The Regents of the University of California. All rights reserved.'


class DQLDaemon:
    """DQLDaemon of ASCAR

    All public members are thread-safe.

    :type controller: DiscreteDeepQ
    :type opt: dict
    :type session: tf.Session

    """
    controller = None
    debugging_level = 0
    delay_between_actions = 1           # seconds between actions
    disable_training = False
    exploration_period = 5000
    opt = None
    last_observation = None
    last_action = None
    new_action = None
    save_path = None
    session = None
    start_random_rate = 0.5
    test_number_of_steps_after_restore = 0
    memcache_last_rowid = 0

    def __init__(self, conf: dict = None, opt: dict = None):
        tf.disable_v2_behavior()
        if 'dqldaemon_debugging_level' in opt:
            self.debugging_level = opt['dqldaemon_debugging_level']

        
        self.stop_requested = False
        self.stopped = True
        self.opt = opt
        self.conf = conf

        self.save_path = os.path.dirname(opt['dbfile'])

        self.memcache = list()

        self.minibatch_size = self.opt['minibatch_size']
        self.ticks_per_observation = self.opt['ticks_per_observation']
        self.disable_training = self.opt['disable_training']
        self.observation_size = self.opt['observation_size']
        # if game:
        #     self.game = game
        # else: # No game assign(Select Lustre game)
        #     from .LustreGame import Lustre
        #     self.opt['disable_same_thread_check'] = True
        #     self.game = Lustre(self.opt, lazy_db_init=True)

        self.db = ReplayDB(self.opt, self.conf)
        self.refresh_memcache()

        # setup tuning system configuration
        if 'delay_between_actions' in opt:
            self.delay_between_actions = opt['delay_between_actions']
        if 'exploration_period' in opt:
            self.exploration_period = opt['exploration_period']
        if 'start_random_rate' in opt:
            self.start_random_rate = opt['start_random_rate']

        self.enable_tuning = self.opt.get('enable_tuning', True)
        self.LOG_DIR = tempfile.mkdtemp()

        print(f"LOG_DIR is locate at {self.LOG_DIR}. To enable Tensorboard run 'tensorboard --logdir [LOG_DIR]'")

    def start(self):
        if self.debugging_level >= 1:
            import cProfile
            import io
            import pstats
            pr = cProfile.Profile()
            pr.enable()
        
        self.stopped = False
        try:
            # self.db.connect_db()

            # TensorFlow business - it is always good to reset a graph before creating a new controller.
            ops.reset_default_graph()
            # TODO: shall we use InteractiveSession()?
            self.session = tf.Session()  # tf.InteractiveSession()

            # This little guy will let us run tensorboard
            #      tensorboard --logdir [LOG_DIR]
            journalist = tf.summary.FileWriter(self.LOG_DIR)

            # Brain maps from observation to Q values for different actions.
            # Here it is a done using a multi layer perceptron with 2 hidden
            # layers
            hidden_layer_size = max(int(self.opt['observation_size'] * 1.2), 200)
            logger.info('Observation size {0}, hidden layer size {1}'.format(self.opt['observation_size'],
                                                                             hidden_layer_size))
            brain = MLP([self.opt['observation_size'], ], [hidden_layer_size, hidden_layer_size,self.opt['num_actions']],
                        [tf.tanh, tf.tanh, tf.identity])

            # The optimizer to use. Here we use RMSProp as recommended
            # by the publication
            optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.9)

            #!!!!
            # DiscreteDeepQ object
            self.controller = DiscreteDeepQ((self.opt['observation_size'],), self.opt['num_actions'], brain, optimizer,
                                            self.session, discount_rate=0.99, start_random_rate=self.start_random_rate,
                                            exploration_period=self.exploration_period,
                                            random_action_probability=self.opt.get('random_action_probability', 0.05),
                                            train_every_nth=1, summary_writer=journalist)

            self.session.run(tf.initialize_all_variables())

            self.session.run(self.controller.target_network_update)

            #checks if there is a model to be loaded before updating the graph
            if os.path.isfile(os.path.join(self.save_path, 'model')):
                self.controller.restore(self.save_path)
                logger.info('Loaded saved model from ' + self.save_path)
            else:
                logger.info('No saved model found')

            self.test_number_of_steps_after_restore = self.controller.actions_executed_so_far



            # graph was not available when journalist was created
            journalist.add_graph(self.session.graph)

            logger.info('DQLDaemon started')

            last_action_second = 0
            last_training_step_duration = 0
            last_checkpoint_time = time.time()
            while not self.stop_requested:
                begin_time = time.time()
                # Start training
                minibatch_size, prediction_error = self._do_training_step()
                logger.info('Finished training step')
                if minibatch_size > 0:
                    if time.time() - last_checkpoint_time > 60*30:
                        # Checkpoint every 30 minutes. TODO: make this a parameter.
                        cp_path = os.path.join(self.save_path, 'checkpoint_' + time.strftime('%Y-%m-%d_%H-%M-%S'))
                        os.mkdir(cp_path)
                        self.controller.save(cp_path)
                        last_checkpoint_time = time.time()
                        logger.info('Checkpoint saved in ' + cp_path)
                    last_training_step_duration = time.time() - begin_time
                    logger.info('Finished {step}th training step in {time} seconds '
                                'using {mb} samples with prediction error {error}.'.format(
                                    step=self.controller.iteration, time=last_training_step_duration, mb=minibatch_size,
                                    error=prediction_error))
                else:
                    logger.info('Not enough data for training yet.')

                # !!!!!!    
                # Always False so it won't be call
                # if self.game.is_over():
                #     logger.info('Game over')
                #     self.stop_requested = True
                #     return

                ts = time.time()
                if ts - (last_action_second+0.5) >= self.delay_between_actions - last_training_step_duration:
                    if self.enable_tuning:
                        try:
                            self.game.refresh_memcache()
                        except:
                            pass

                        sleep_time = max(0, self.delay_between_actions - (time.time() - (last_action_second + 0.5)))
                        if sleep_time > 0.05:
                            # Do cleaning up before long sleeping
                            gc.collect()
                            sleep_time = max(0, self.delay_between_actions - (time.time() - (last_action_second + 0.5)))
                        if sleep_time > 0.0001:
                            logger.debug('Sleeping {0} seconds'.format(sleep_time))
                            time.sleep(sleep_time)
                        self._do_action_step()
                        last_action_second = int(time.time())
                    else:
                        logger.debug('Tuning disabled.')
                        # Check for new data every 200 steps to reduce checking overhead
                        if self.controller.number_of_times_train_called % 200 == 0:
                            try:
                                self.game.refresh_memcache()
                                pass
                            except:
                                pass
                    # We always print out the reward to the log for analysis
                    # !!!logger.info('Cumulative reward: {0}'.format(self.game.cumulative_reward))

                    flush_log()
        finally:
            self.stopped = True
            # controller.save should not work here as the controller is still NoneType
            #self.controller.save(self.save_path)
            logger.info('DQLDaemon stopped. Model saved in ' + self.save_path)

            if self.debugging_level >= 1:
                pr.disable()
                s = io.StringIO()
                sortby = 'cumulative'
                ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                ps.print_stats()
                print(s.getvalue())
        
        self.db.conn.close()

    def _do_training_step(self) -> (int, float):
        """ Do a training step

        This function is NOT thread-safe and can only be called within the worker thread.

        :return: size of the mini batch, prediction error
        """
        if not self.disable_training:
            mini_batch = self.get_minibatch()
            logger.info(f'Retrieve batch size: {len(mini_batch)}{mini_batch}')
            if mini_batch:
                logger.debug('Got minibatch of size {0}'.format(len(mini_batch)))
                return len(mini_batch), self.controller.training_step(mini_batch)
            else:
                return 0, None
        else:
            raise RuntimeError('Training is disabled')

    def _do_action_step(self):
        """ Do an action step

        This function is NOT thread-safe and can only be called within the worker thread.

        :return:
        """
        if not self.enable_tuning:
            raise RuntimeError('Tuning is disabled')

        try:
            new_observation = self.observe()
            reward = self.collect_reward()
            pass
        except BaseException as e:
            logger.info('{0}. Skipped taking action.'.format(str(e)))
            traceback.print_exc()
            return

        # Store last transition. This is only needed for the discrete hill test case.
        # NOTIMPLEMENT: CAPES doesn't implement this function yet
        # if self.last_observation is not None:
        #     self.store(self.last_observation, self.last_action, reward, new_observation)
        #     pass
        # act
        self.new_action = self.controller.action(new_observation)
        self.perform_action(self.new_action)

        self.last_action = self.new_action
        self.last_observation = new_observation
        

    def is_game_over(self) -> bool:
        """Check if the game is over

        This function is thread-safe

        :return: A bool that represents if the game is over
        """
        # First check if the game is stopped, if not we can't safely read self.game
        if not self.is_stopped():
            return False

        # !!!return self.game.is_over()

    def is_stopped(self) -> bool:
        """ Check if the worker thread is stopped

        This function is thread-safe.

        :return:
        """
        return self.stopped

    def join(self):
        while not self.stopped:
            time.sleep(0.2)

    def stop(self):
        """ Stop the daemon

        This function is thread-safe.
        :return:
        """
        self.stop_requested = True
        logger.info('Requesting DQLDaemon to stop...')

    # ================= Get it from LustreGame.py =========================

    def refresh_memcache(self):
        logger.info('Loading cache...')
        c = self.db.conn.cursor()
        # Use a large arraysize to increase read speed; we don't care about memory usage
        c.arraysize = 1000000
        if not self.memcache:
            self.memcache = list()
        preloading_cache_size = len(self.memcache)
        # Getting all data from perf and actions
        c.execute('''SELECT perfs.rowid, perfs.clientid, perfs.ts, perfs.pis, action
                    FROM perfs LEFT JOIN actions ON perfs.ts=actions.ts
                    WHERE perfs.rowid > ? ORDER BY perfs.ts, perfs.clientid''',
                (self.memcache_last_rowid,))
        f = c.fetchall()
        # For each row
        for row in f:
            #set last row
            self.memcache_last_rowid = max(self.memcache_last_rowid, row[0])
            # add id and pis from query data
            clientid, ts, pi_data = row[1], row[2], pickle.loads(row[3])
            # also for action
            action = pickle.loads(row[4])

            

            # if clientid not in self.db.client_list:
            #     continue
            # assert len(pi_data) == self.db.tick_data_size // len(self.db.ordered_client_list)

            if len(self.memcache) == 0 or self.memcache[-1][0] != ts:
                self.memcache.append((ts, action, [None] * len(self.db.client_list)))
            self.memcache[-1][2][self.db.client_list.index(clientid)] = np.array(pi_data)

        # Peak memory usage (bytes on OS X, kilobytes on Linux)
        # https://stackoverflow.com/a/7669482
        logger.info('Finished loading {len} entries. Peak memory usage {size:,}.'.format(
            len=len(self.memcache) - preloading_cache_size,
            size=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

    def get_minibatch(self):
        # We need at least ticks_per_observation+1 ticks for one sample
        if len(self.memcache) < self.ticks_per_observation + 1:
            return None
        good_idx = set()
        result = []
        required_samples = self.minibatch_size
        while True:
            # total possible sample size after removing the first (ticks_per_observation-1) ticks
            # and the last tick (because we need ts+1 in the sample)
            total_sample_size = len(self.memcache) - self.ticks_per_observation
            if total_sample_size <= len(result):
                return result
            required_samples = min(total_sample_size, required_samples)
            # The last idx has to be excluded so it won't be added to bad_idx set
            samples = random.sample(range(self.ticks_per_observation - 1, len(self.memcache)-1),
                                    required_samples - len(result))
            for i in samples:
                if i in good_idx:
                    continue
                try:
                    # get observation and next
                    observ = self.get_observation_by_cache_idx(i)
                    observ_next = self.get_next_observation_by_cache_idx(i)

                    # calculate reward
                    reward = self._calc_total_throughput(observ_next) - self._calc_total_throughput(observ)
                    # The final ts is only used in test cases
                    ts = self.memcache[i][0]
                    result.append((observ, self.memcache[i][1], reward, observ_next, ts))

                    good_idx.add(ts)
                    if len(result) == required_samples:
                        logger.info(result)
                        return result
                except NotEnoughDataError:
                    logger.info('NotEnoughDataError for memcache idx {0}'.format(i))

    def observe(self) -> np.ndarray:
        """Return observation vector using memcache.
        """
        err_msg = 'No valid observation in the past two seconds'
        if len(self.memcache) < self.ticks_per_observation:
            raise NotEnoughDataError(err_msg)
        for idx in range(len(self.memcache)-1, max(len(self.memcache)-3, self.ticks_per_observation-1), -1):
            try:
                return self.get_observation_by_cache_idx(idx)
            except NotEnoughDataError:
                pass
        raise NotEnoughDataError(err_msg)

    def collect_reward(self):
        """Reward is the sum of read throughput + write throughput of clients
        """
        o, prevo = self.db.get_last_n_observation(2)
        return self._calc_total_throughput(o) - self._calc_total_throughput(prevo)

    def perform_action(self, action_id: int):
        """Send the new action to IntfDaemon
        """
        assert 0 <= action_id < self.num_actions

        if not self.cpvs:
            # use the default value
            self.cpvs = [x[1] for x in self.opt['cpvs']]

        if action_id > 0:
            cpv_id = (action_id - 1) // 2
            lower_range = self.opt['cpvs'][cpv_id][2]
            upper_range = self.opt['cpvs'][cpv_id][3]
            step = self.opt['cpvs'][cpv_id][4]
            if action_id % 2 == 0:
                # minus step
                if self.cpvs[cpv_id] < lower_range + step:
                    # invalid move, do nothing
                    pass
                else:
                    self.cpvs[cpv_id] -= step
            else:
                # plus 1
                if self.cpvs[cpv_id] > upper_range - step:
                    # invalid move, do nothing
                    pass
                else:
                    self.cpvs[cpv_id] += step

        # Broadcast action must begin with action_id, which will be saved by
        # IntfDaemon to the DB.
        ControllerIntf.broadcast_action([action_id] + self.cpvs)

    def cumulative_reward(self):
        try:
            return self._calc_total_throughput(self.db.get_last_n_observation()[0])
        except NotEnoughDataError:
            return 0

    def _calc_total_throughput(self, o: np.ndarray) -> float:
        """Calculate total throughput of an observation

        Only the throughput of last tick in the observation are included in the reward.
        :param o: observation
        :return: total throughput
        """
        if 'client_id' in self.opt:
            num_ma = len(self.opt['client_id'])
        else:
            num_ma = 1
        # reshape also checks the shape of o
        o = np.reshape(o, (num_ma, self.ticks_per_observation,
                        2))
        result = 0.0
        for ma_id in range(num_ma):
            for osc in range(2):
                read_tp_ix  = 0
                write_tp_ix = 0


                logger.info(f'print this: {o[ma_id, self.ticks_per_observation-1, read_tp_ix]}')
                read_tp  = o[ma_id, self.ticks_per_observation-1, read_tp_ix]
                write_tp = o[ma_id, self.ticks_per_observation-1, write_tp_ix]
                # sanity check: our machine can't be faster than 300 MB/s
                assert 0 <= read_tp <= 300 * 1024 * 1024
                assert 0 <= write_tp <= 300 * 1024 * 1024
                result += read_tp + write_tp
        return result

        def get_minibatch_from_db(self):
            good_ts = set()
            bad_ts = set()
            result = []
            required_samples = self.minibatch_size
            while True:
                try:
                    # DB data may change during our run so we query it every time
                    min_ts, max_ts = self.db.get_action_ts_range()
                    pi_min_ts, pi_max_ts = self.db.get_pi_ts_range()
                except NotEnoughDataError:
                    return None
                # we need at least ticks_per_observation+1 ticks for one sample
                if (pi_max_ts - pi_min_ts < self.ticks_per_observation) or \
                (max_ts - min_ts < self.ticks_per_observation):
                    return None
                # Calculate the starting ts after which we can take valid observation samples
                min_ts = max(min_ts, pi_min_ts + self.ticks_per_observation - 1)
                # total possible sample size after removing the first (ticks_per_observation-1) ticks
                # and the last tick (because we need ts+1 in the sample)
                total_sample_size = max_ts - min_ts - len(bad_ts)
                if total_sample_size <= len(result):
                    return result
                required_samples = min(total_sample_size, required_samples)
                samples = random.sample(range(min_ts, max_ts), required_samples - len(result))
                for ts in samples:
                    if ts in good_ts or ts in bad_ts:
                        continue
                    try:
                        observ = self.db.get_observation(ts)
                        observ_next = self.db.get_observation(ts + 1)
                        reward = self._calc_total_throughput(observ_next) - self._calc_total_throughput(observ)
                        # The final ts is only used in test cases
                        result.append((observ, self.db.get_action(ts), reward, observ_next, ts))

                        good_ts.add(ts)
                        if len(result) == required_samples:
                            self.TestSample = list(good_ts)
                            return result
                    except NotEnoughDataError:
                        logger.warning('NotEnoughDataError for ts {0}'.format(ts))
                        bad_ts.add(ts)

    def get_observation_by_cache_idx(self, idx: int) -> np.ndarray:
        assert 0 <= idx < len(self.memcache)
        if idx < self.ticks_per_observation - 1:
            raise NotEnoughDataError
        # Return None if the time is not continuous
        idx_start = idx - self.ticks_per_observation + 1
        if self.memcache[idx][0]- self.memcache[idx_start][0] < 9:
            raise NotEnoughDataError('Not enough tick data')

        result = np.zeros((len(self.db.client_list),
                           self.ticks_per_observation,
                           int(self.db.tick_data_size / len(self.db.client_list))), dtype=float)
        missing_entry = 0

        for i in range(idx_start, idx+1):
            for client_id_idx in range(len(self.db.client_list)):
                if self.memcache[i][2][client_id_idx] is None:
                    missing_entry += 1
                    if missing_entry > self.db.missing_entry_tolerance:
                        raise NotEnoughDataError('Too many missing entries')
                else:
                    logger.info(f'Read IO_bytes: {self.memcache[i][2][client_id_idx]}')
                    piDict = str(self.memcache[i][2][client_id_idx])
                    piDict = piDict.replace("'",'"')
                    piDict = json.loads(piDict)
                    result[client_id_idx, i-idx_start] = (piDict['read']['io_bytes'], piDict['write']['io_bytes'])
        return result.reshape((self.observation_size,))

    def get_next_observation_by_cache_idx(self, idx: int) -> np.ndarray:
        assert 0 <= idx < len(self.memcache)
        # if idx == len(self.memcache) - 1:
        #     raise NotEnoughDataError
        # if self.memcache[idx][0] + 1 != self.memcache[idx + 1][0]:
        #     raise NotEnoughDataError
        # print("Hello world3")
        # logger.info('Hello world3')
        return self.get_observation_by_cache_idx(idx + 1)