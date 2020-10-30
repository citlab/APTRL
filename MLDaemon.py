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
import time
import traceback
import resource
import random
import json
import tempfile

# Tensorflow stuff here
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops

# Local module stuff here
from ascar_logging import *
from tf_rl.controller import DiscreteDeepQ
from tf_rl.models import MLP
from ReplayDB import *
from ControllerIntf import ControllerInft

import sqlite3

__author__ = 'Yan Li'
__copyright__ = 'Copyright (c) 2016, 2017 The Regents of the University of California. All rights reserved.'

class MLDaemon:
    """DQLDaemon of ASCAR

    All public members are thread-safe.

    :type controller: DiscreteDeepQ
    :type opt: dict
    :type conf: dict
    :type session: tf.Session
    """
    
    controller = None                   # Tensorflow controller for MLP
    session = None                      # Tensorflow session


    debugging_level = 0                 # debug level for log
    disable_training = False            # training default is enable
    enable_tuning = False               # tuning default is set to disable
    stop_requested = False              # stop training and action deciding
    stopped = True                      # stop daemon
    
    opt = None                          # option configuration for deep learning
    conf = None                         # all configurations

    delay_between_actions = 1           # seconds between actions
    exploration_period = 5000           # exploration interval in one period
    start_random_rate = 0.5             # random rate for selecting samples
    checkpoint_time = 1800              # time of saving each checkpoint

    last_observation = None             # last observation used in training process
    last_action = None                  # last set of parameter configurations
    last_step = None                    # last set of parameter step size
    new_action = None                   # new action that will be broadcast to storage system
    save_path = None                    # save path of model and log file

    cumulative_reward = 0               # cumulative rewards that Deep Q learning get at the end

    test_number_of_steps_after_restore = 0  # number of test steps after restore tensorflow model
    memcache_last_rowid = 0                 # lastest row of memcache

    def __init__(self, conf: dict = None, opt: dict = None):
        # TODO: Change to version 2 when finish everything
        tf.disable_v2_behavior()       # disable tensorflow version 2

        # get debugging level from config file
        if 'dqldaemon_debugging_level' in conf['log']:
            self.debugging_level = conf['log']['dqldaemon_debugging_level'] 

        # assign configuration and option
        self.opt = opt
        self.conf = conf

        # get directory for saving model and log_dir
        self.save_path = os.path.dirname(conf['replaydb']['dbfile'])

        self.disable_training = self.opt['disable_training']            # get disable_training option from config file 

        self.minibatch_size = self.opt['minibatch_size']                # mini batch size for training in one observation (Size of PIs in one row of pi db)
        self.ticks_per_observation = self.opt['ticks_per_observation']  # Number of Ticks (depends of tick_len) per 1 observation
        self.observation_size = self.opt['observation_size']            # Size of one observation

        # setup tuning system configuration (Description is at the beginning)
        if 'delay_between_actions' in opt:
            self.delay_between_actions = opt['delay_between_actions']
        if 'exploration_period' in opt:
            self.exploration_period = opt['exploration_period']
        if 'start_random_rate' in opt:
            self.start_random_rate = opt['start_random_rate']
        if 'checkpoint_time' in opt:
            self.checkpoint_time = opt['checkpoint_time']

        self.enable_tuning = self.opt['enable_tuning']

        # Initialize database and retrieve data from database
        self.db = ReplayDB(self.opt, self.conf)
        self.db.refresh_memcache()
        
        # Store default action
        default = []
        default_step = []
        for param in self.conf['ceph-param']:
            val = list(param.values())[0]
            default.append(val['default'])
            default_step.append(val['step'])
        self.last_action = default
        self.last_step = default_step
        
        # make temp file for storing tensorflow log
        self.LOG_DIR = tempfile.mkdtemp()
        logger.info(f"LOG_DIR is locate at {self.LOG_DIR}. To enable Tensorboard run 'tensorboard --logdir [LOG_DIR]'")

    def start(self):
        """Start MLDaemon
        
        This function create tensorflow controller and running the tuning by iteratively 
        training and choose action.
        """
        if self.debugging_level >= 1:
            import cProfile
            import io
            import pstats
            pr = cProfile.Profile()
            pr.enable()

        logger.info(f"Connected to database {self.conf['replaydb']['dbfile']}")
        
        # set stopped to False, so daemon can run
        self.stopped = False

        logger.info('Starting MLDaemon...')
        try:
            # TensorFlow business - it is always good to reset a graph before creating a new controller.
            ops.reset_default_graph()
            # ? shall we use InteractiveSession()?
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
            # DiscreteDeepQ object
            # TODO: change num_actions and other configuration
            self.controller = DiscreteDeepQ((self.opt['observation_size'],), self.opt['num_actions'], brain, optimizer,
                                            self.session, discount_rate=0.99, start_random_rate=self.start_random_rate,
                                            exploration_period=self.exploration_period,
                                            random_action_probability=self.opt['random_action_probability'],
                                            train_every_nth=1, summary_writer=journalist, k_action=int(self.opt['k_val']))

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

            last_action_second = 0              # last action timestep
            last_training_step_duration = 0     # last training duration
            last_checkpoint_time = time.time()  # last checkpoint
            while not self.stop_requested:
                begin_time = time.time()        # set begin time to current time

                # Run training step
                logger.info('Start training step...')
                minibatch_size, prediction_error = self._do_training_step()

                if minibatch_size > 0:
                    # Check checkpoint time for every self.checkpoint_time
                    logger.info(f'Time before checkpoint: {self.checkpoint_time - (time.time() - last_checkpoint_time)}')
                    if time.time() - last_checkpoint_time > self.checkpoint_time:
                        # save controller checkpoint
                        cp_path = os.path.join(self.save_path, 'checkpoint_' + time.strftime('%Y-%m-%d_%H-%M-%S'))
                        os.mkdir(cp_path)
                        self.controller.save(cp_path)
                        # update checkpoint time
                        last_checkpoint_time = time.time()
                        logger.info('Checkpoint saved in ' + cp_path)

                    # update last training duration
                    last_training_step_duration = time.time() - begin_time
                    logger.info('Finished {step}th training step in {time} seconds '
                                'using {mb} samples with prediction error {error}.'.format(
                                    step=self.controller.iteration, time=last_training_step_duration, mb=minibatch_size,
                                    error=prediction_error))
                else:
                    logger.info('Not enough data for training yet.')

                # Check if it is time for tuning
                # (check if duration since last action passed compare to time left before next actions)
                if time.time() - (last_action_second+0.5) >= self.delay_between_actions - last_training_step_duration:
                    if self.enable_tuning:
                        logger.debug('Start tuning step...')

                        try:
                            # Update memcache for next traininf interval
                            self.db.refresh_memcache()
                        except:
                            pass

                        # get sleep time either 0 or what is left until next action is start
                        sleep_time = max(0, self.delay_between_actions - (time.time() - (last_action_second + 0.5)))
                        if sleep_time > 0.05:
                            # Do garbage cleaning up before long sleeping
                            gc.collect()
                            sleep_time = max(0, self.delay_between_actions - (time.time() - (last_action_second + 0.5)))
                        if sleep_time > 0.0001:
                            logger.debug(f'Sleeping {sleep_time} seconds')
                            # Welp, basically sleep
                            time.sleep(sleep_time)
                        
                        # Do action step
                        ts = int(time.time())
                        self._do_action_step(ts)
                        # Update action to current time
                        last_action_second = ts
                    else:
                        logger.debug('Tuning disabled.')
                        # Check for new data every 200 steps to reduce checking overhead
                        if self.controller.number_of_times_train_called % 200 == 0:
                            try:
                                self.db.refresh_memcache()
                                pass
                            except:
                                pass

                    # We always print out the reward to the log for analysis
                    logger.info(f'Cumulative reward: {self.cumulative_reward}')
                    
                    # Clean log at the end for next run
                    flush_log()
        finally:
            # set stopped to True, so daemon can properly stop 
            self.stopped = True
            # controller.save should not work here as the controller is still NoneType
            # self.controller.save(self.save_path)
            logger.info('DQLDaemon stopped.')

            if self.debugging_level >= 1:
                pr.disable()
                s = io.StringIO()
                sortby = 'cumulative'
                ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                ps.print_stats()
                print(s.getvalue())
                
    @staticmethod
    def store(*unused):
        pass

    def _do_training_step(self) -> (int, float):
        """Do a training step
        
        This function is NOT thread-safe and can only be called within the worker thread.
        
        Raises:
            RuntimeError: Training is set to disable
            
        Returns:
            (int, float): size of the mini batch, prediction error
        """
        if not self.disable_training:
            # Get training batch from memcache in replay database
            mini_batch = self.get_minibatch()
            if mini_batch:
                logger.info(f'Retrieve batch size: {len(mini_batch)}')
                return len(mini_batch), self.controller.training_step(mini_batch)
            else:
                return 0, None
        else:
            raise RuntimeError('Training is disabled')


    def _do_action_step(self, ts):
        """ Do an action step

        This function is NOT thread-safe and can only be called within the worker thread.

        Raises:
            RuntimeError: Tuning is disable, so no action will be perform
        """
        if not self.enable_tuning:
            raise RuntimeError('Tuning is disabled')

        try:
            # get new observation
            new_observation = self.observe()
            # collect reward
            reward = self.collect_reward()
        except BaseException as e:
            logger.info('{0}. Skipped taking action.'.format(str(e)))
            traceback.print_exc()
            return

        # Store last transition. This is only needed for the discrete hill test case.
        if self.last_observation is not None:
            # TODO: Implement store function (CAPES haven't implement this function yet)
            self.store(self.last_observation, self.last_action, reward, new_observation)
            pass
        # get action from new observation
        self.new_action = self.controller.action(new_observation)
        # Perform action
        self.perform_action(self.new_action, ts)

        # Update last observation to current one
        self.last_observation = new_observation


    def get_minibatch(self):
        """Get mini batch for training
        
        This function is NOT thread-safe and can only be called within the worker thread.
        It calls ReplayDB to retrieve mini batch

        Returns:
            list: mini batch containing timestep, action, reward of the action, observation of 
        a current timestep, and next observation
        """
        # We need at least ticks_per_observation+1 ticks for one sample
        if len(self.db.memcache) < self.ticks_per_observation + 1:
            return None

        good_idx = set()    # TODO: Should I use this?
        result = [] # mini batch that will be return in the end
        required_samples = self.minibatch_size  # samples per batch
        while True:
            # Get total sample size available after removing the first ticks_per_observation-1 ticks
            total_sample_size = len(self.db.memcache) - self.ticks_per_observation  # TODO: if using good_idx and bad_idx this line must subtract by bad idx as well
            # If possible sample size is not enough
            if total_sample_size <= len(result):
                return result
            # get required sample from least size
            required_samples = min(total_sample_size, required_samples)
            # The last idx has to be excluded so it won't be added to bad_idx set (last idx 
            # is not yet decided on action)
            # pick sample index at random from tick_per_observation -1 ticks to last ticks 
            # with size of remaining required sample
            sample_idx = random.sample(range(self.ticks_per_observation - 1, len(self.db.memcache)-1),
                                    required_samples - len(result))
            for i in sample_idx:
                # TODO: Should I use this?
                if i in good_idx:
                    continue
                try:
                    # get observation and next observation from sample index
                    observ = self.get_observation_by_cache_idx(i)
                    observ_next = self.get_next_observation_by_cache_idx(i)

                    # calculate reward from observation of current step and next step
                    reward = self._calc_total_throughput(observ_next) - self._calc_total_throughput(observ)
                    
                    # The final ts is only used in test cases
                    # append data into result as tuple
                    ts = self.db.memcache[i][0]
                    action = self.db.memcache[i][1]
                    # Prevent getting observation without action append to training batch
                    if(action != [-1]):
                        result.append((observ, action, reward, observ_next, ts))

                    # TODO: should I use this?
                    good_idx.add(ts)
                    if len(result) == required_samples:
                        logger.debug(f'Retrieved mini batach with data: {result}')
                        return result
                except NotEnoughDataError:
                    logger.info(f'NotEnoughDataError for memcache idx {i}')
    
    def observe(self) -> np.ndarray:
        """ Return lastest observation vector using memcache.
        
        Get observation vector using memcache

        Raises:
            NotEnoughDataError: ticks in memcache is not enough
            NotEnoughDataError: cannot get observation from 
                self.get_observation_by_cache_idx(idx)

        Returns:
            np.ndarray: Observation at index idx
        """
        
        err_msg = 'No valid observation in the past two seconds'
        
        # If ticks in memcache is not enough
        if len(self.db.memcache) < self.ticks_per_observation:
            raise NotEnoughDataError(err_msg)
        # Loop from last ticks (last memcache index) to at least third last ticks or ticks_per_observation tick
        for idx in range(len(self.db.memcache)-1, max(len(self.db.memcache)-3, self.ticks_per_observation-1), -1):
            try:
                # Return lastest possible observation
                return self.get_observation_by_cache_idx(idx)
            except NotEnoughDataError:
                pass
        # No observation found, so raise with err_msg
        raise NotEnoughDataError(err_msg)

    def collect_reward(self):
        """ Reward is the sum of read throughput + write throughput of clients

        Return reward for the last observation
        
        Returns:
            float: reward from current total thoughput and previous total throughput
        """
        observ, prev_observ = self.db.get_last_n_observation(2)
        return self._calc_total_throughput(observ) - self._calc_total_throughput(prev_observ)

    def get_observation_by_cache_idx(self, idx: int) -> np.ndarray:
        """ Get observation from index

        Args:
            idx (int): observation index

        Raises:
            NotEnoughDataError: Index is greater than tick_per_observation
            NotEnoughDataError: Index does not belong to the same observation
            NotEnoughDataError: Missing entry exceed the missing_entry_tolerance

        Returns:
            np.ndarray: observation data of the index
        """
        
        # Check if idx is out of range
        assert 0 <= idx < len(self.db.memcache)
        
        # Check if index is consider in the observation 
        if idx < self.ticks_per_observation - 1:
            raise NotEnoughDataError
        # Return None if the time is not continuous
        idx_start = idx - self.ticks_per_observation + 1
        # if self.db.memcache[idx_start][0] != (self.db.memcache[idx][0] * self.opt['tick_len']) + 1:
        #     raise NotEnoughDataError('Not enough tick data')

        # Create result nd.array for storing data from memcache
        # With client_len, number of tick per observation, data per client
        result = np.zeros((len(self.db.client_list),
                        self.ticks_per_observation,
                        int(self.db.tick_data_size / len(self.db.client_list))), dtype=float)
        missing_entry = 0

        # Loop each index until reach idx
        for i in range(idx_start, idx+1):
            # Loop each client
            for client_id_idx in range(len(self.db.client_list)):
                # Check for missing entry
                if self.db.memcache[i][2][client_id_idx] is None:
                    missing_entry += 1  # Add missing entry
                    # Check for tolerance
                    if missing_entry > self.db.missing_entry_tolerance:
                        raise NotEnoughDataError('Too many missing entries')
                else:
                    # Add PIs data at i to result list
                    result[client_id_idx, i-idx_start] = self.db.memcache[i][2][client_id_idx]
        # return result vector
        return result.reshape((self.observation_size,))

    def get_next_observation_by_cache_idx(self, idx: int) -> np.ndarray:
        """Get next observation from idx

        Args:
            idx (int): observation index

        Raises:
            NotEnoughDataError: index is last tick in memcache
            NotEnoughDataError: next tick is not equal to ts at idx + tick_len

        Returns:
            np.ndarray: next observation
        """
    
        # Check if idx is out of range
        assert 0 <= idx < len(self.db.memcache)
        # if index is the last tick
        if idx == len(self.db.memcache) - 1:
            raise NotEnoughDataError
        # Check if next tick is continuous
        if self.db.memcache[idx][0] + self.opt['tick_len'] != self.db.memcache[idx + 1][0]:
            raise NotEnoughDataError
        # Check observation at next tick
        return self.get_observation_by_cache_idx(idx + 1)
    
    def _calc_total_throughput(self, observ: np.ndarray) -> float:
        """ Calculate total throughput of an observation

        Only the throughput of last tick in the observation are included in the reward.

        Args:
            observ (np.ndarray): observation

        Returns:
            float: total throughput
        """
        # Get number of client
        if 'client_id' in self.conf['node']:
            client_num = len(self.conf['node']['client_id'])
        else:
            client_num = 1
            
        # reshape also checks the shape of observ
        observ = np.reshape(observ, (client_num, self.ticks_per_observation,
                        int(self.db.tick_data_size / len(self.db.client_list))))
        result = 0.0        # throughput result
        
        # Loop through each client
        for client_idx in range(client_num):
            # Loop through each server (Should start from 0)
            for osc in range(len(self.conf['node']['server_id'])):
                # Get read_bytes, write_bytes indicies
                read_ix  = osc * (self.opt['pi_per_client_obd']) + 0
                write_ix = osc * (self.opt['pi_per_client_obd']) + 1
                # Get read_bytes, writes_bytes from observation
                read_bytes  = observ[client_idx, self.ticks_per_observation-1, read_ix]
                write_bytes = observ[client_idx, self.ticks_per_observation-1, write_ix]
                # sanity check: our machine can't be faster than 300 MB/s
                assert 0 <= read_bytes <= 300 * 1024 * 1024
                assert 0 <= write_bytes <= 300 * 1024 * 1024
                result += read_bytes + write_bytes
        return result
    
    def _calc_total_latency(self, observ: np.ndarray) -> float:
        """ Calculate latency of an observation

        Only the lantency of last tick in the observation are included in the reward.

        Args:
            observ (np.ndarray): observation

        Returns:
            float: total throughput
        """
        # Get number of client
        if 'client_id' in self.opt:
            client_num = len(self.opt['client_id'])
        else:
            client_num = 1
            
        # reshape also checks the shape of observ
        observ = np.reshape(observ, (client_num, self.ticks_per_observation,
                        int(self.db.tick_data_size / len(self.db.client_list))))
        result = 0.0        # throughput result
        
        # Loop through each client
        for client_idx in range(client_num):
            # Loop through each server
            for osc in range(len(self.opt['server_id'])):
                # Get latency read and write indicies
                latency_r_ix  = osc * (self.opt['pi_per_client_obd']) + 2
                latency_w_ix = osc * (self.opt['pi_per_client_obd']) + 3 

                # Get latency read and write from observation
                latency_r  = observ[client_num, self.ticks_per_observation-1, latency_r_ix]
                latency_w = observ[client_num, self.ticks_per_observation-1, latency_w_ix]
                result += latency_r + latency_w
        return result
    
    def perform_action(self, actions, ts):
        """Send the new action to IntfDaemon

        Args:
            action: An action predicted by discrete_deepq
        """
        # logger.info(f'{action}')
        # assert action.shape == tuple([1,4])
        # assert 0 <= action_id < self.opt['num_actions']
            
        # TODO: check increase, decrease or nothing  
        # TODO: store action in replayDB and then broadcast param to Storage
        for action_id in actions:
            if action_id > 0:
                param_id = (action_id-1) // 4
                param_valu = list(self.conf['ceph-param'][param_id].values())[0]
                param_type = param_valu['type']
                min_val = param_valu['min']
                max_val = param_valu['max']
                step_change = self.opt['stepsize_change']
                if action_id % 2 == 0:
                    # minus step
                    if self.last_action[param_id] - self.last_step[param_id] < min_val:
                        # invalid move
                        pass
                    else:
                        self.last_step[param_id] -= step_change
                        self.last_action[param_id] -= self.last_step[param_id]
                else:
                    # plus step
                    if self.last_action[param_id] + self.last_step[param_id] > max_val:
                        # invalid move
                        pass
                    else:
                        self.last_step[param_id] += step_change
                        self.last_action[param_id] += self.last_step[param_id]
                    

        self.db.connect_db()
        for t in (ts-self.delay_between_actions, ts):
            try:
                logger.info(f'insert action at: {t}')
                self.db.insert_action(t, self.last_action)
            except sqlite3.IntegrityError as e:
                pass
        self.db.conn.close()
        # # Broadcast action must begin with action_id, which will be saved by
        # # IntfDaemon to the DB.
        ControllerInft.broadcastAction(self.last_action, ts, self.conf, self.opt)