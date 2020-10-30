"""ASCAR Discreet Deep Q Controller.

The following code are based on https://github.com/nivwusquorum/tensorflow-deepq under
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

Modified by Yan Li <yanli@tuneup.ai>, Kenneth Chang <kchang44@ucsc.edu>,
Oceane Bel <obel@ucsc.edu>. Storage Systems Research Center, Baskin School
of Engineering. The new code is under the following license:

Copyright (c) 2016, 2017 The Regents of the University of California. All
rights reserved.

Created by Yan Li <yanli@tuneup.ai>, Kenneth Chang <kchang44@ucsc.edu>,
Oceane Bel <obel@ucsc.edu>. Storage Systems Research Center, Baskin School
of Engineering.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
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
"""


from ascar_logging import logger
import numpy as np
import random
import tensorflow.compat.v1 as tf
import os
import pickle
import time
#from APTRL.ascar_logging import logger


class DiscreteDeepQ(object):
    def __init__(self, observation_shape,
                    num_actions,
                    observation_to_actions,
                    optimizer,
                    session,
                    random_action_probability=0.05,
                    exploration_period=1000,
                    train_every_nth=1,
                    discount_rate=0.95,
                    start_random_rate = 0.5092,
                    target_network_update_rate=0.01,
                    summary_writer=None,
                    k_action=1):
        """Initialized the Deepq object.

        Based on:
            https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

        Parameters
        -------
        observation_shape : int
            length of the vector passed as observation
        num_actions : int
            number of actions that the model can execute
        observation_to_actions: dali model
            model that implements activate function
            that can take in observation vector or a batch
            and returns scores (of unbounded values) for each
            action for each observation.
            input shape:  [batch_size] + observation_shape
            output shape: [batch_size, num_actions]
        optimizer: tf.solver.*
            optimizer for prediction error
        session: tf.Session
            session on which to execute the computation
        random_action_probability: float (0 to 1)
        exploration_period: int
            probability of choosing a random
            action (epsilon form paper) annealed linearly
            from 1 to random_action_probability over
            exploration_period
        store_every_nth: int
            to further decorrelate samples do not all
            transitions, but rather every nth transition.
            For example if store_every_nth is 5, then
            only 20% of all the transitions is stored.
        train_every_nth: int
            normally training_step is invoked every
            time action is executed. Depending on the
            setup that might be too often. When this
            variable is set set to n, then only every
            n-th time training_step is called will
            the training procedure actually be executed.
        minibatch_size: int
            number of state,action,reward,newstate
            tuples considered during experience reply
        dicount_rate: float (0 to 1)
            how much we care about future rewards.
        max_experience: int
            maximum size of the reply buffer
        target_network_update_rate: float
            how much to update target network after each
            iteration. Let's call target_network_update_rate
            alpha, target network T, and network N. Every
            time N gets updated we execute:
                T = (1-alpha)*T + alpha*N
        summary_writer: tf.train.SummaryWriter
            writer to log metrics
        """
        
        '''
        
        observation_shape = (self.opt['observation_size'],), 
        num_actions = self.opt['num_actions'], 
        observation_to_actions = brain, 
        optimizer = optimizer,
        session = self.session, 
        discount_rate=0.99,
        start_random_rate=self.start_random_rate,
        exploration_period=self.exploration_period,
        random_action_probability=self.opt['random_action_probability'],
        train_every_nth=1, 
        summary_writer=journalist
        
        '''
        
        # memorize arguments
        self.observation_shape         = observation_shape
        self.num_actions               = num_actions

        self.start_random_rate         = start_random_rate      # Beginning random rate

        self.q_network                 = observation_to_actions # MLP
        self.optimizer                 = optimizer
        self.s                         = session

        self.random_action_probability = random_action_probability
        self.exploration_period        = exploration_period
        self.train_every_nth           = train_every_nth
        self.discount_rate             = tf.constant(discount_rate)
        self.target_network_update_rate = \
                tf.constant(target_network_update_rate)

        # deepq state
        self.actions_executed_so_far = 0

        self.iteration = 0
        self.summary_writer = summary_writer

        self.number_of_times_train_called = 0
        
        self.k_action = k_action

        self.create_variables()

        self.s.run(tf.initialize_all_variables())
        self.s.run(self.target_network_update)

        self.saver = tf.train.Saver()

    @staticmethod
    def linear_annealing(n, total, p_initial, p_final):
        """Linear annealing between p_initial and p_final
        over total steps - computes value at step n"""
        if n >= total:
            return p_final
        else:
            return p_initial - (n * (p_initial - p_final)) / total

    def observation_batch_shape(self, batch_size):
        return tuple([batch_size] + list(self.observation_shape))

    def create_variables(self):
        # Create copy for MLP
        self.target_q_network    = self.q_network.copy(scope="target_network")

        # FOR REGULAR ACTION SCORE COMPUTATION
        with tf.name_scope("taking_action"):
            # Create placeholder for accepting array of float with size None, observation_shape
            self.observation        = tf.placeholder(tf.float32, self.observation_batch_shape(None), name="observation")
            # create tensor with same size and content as hidden layers
            self.action_scores      = tf.identity(self.q_network(self.observation), name="action_scores")
            tf.summary.histogram("action_scores", self.action_scores)
            # index of maximum value in MLP output with current observation feed into it
            # self.predicted_actions  = tf.identity(self.action_scores, name="predicted_actions")
            self.predicted_actions = tf.identity(self.action_scores, name="predicted_actions")
            
        with tf.name_scope("estimating_future_rewards"):
            # FOR PREDICTING TARGET FUTURE REWARDS
            # Create placeholder for next observation
            self.next_observation          = tf.placeholder(tf.float32, self.observation_batch_shape(None), name="next_observation")
            # TODO: What is this?
            self.next_observation_mask     = tf.placeholder(tf.float32, (None,), name="next_observation_mask")
            # TODO: What is this?
            self.next_action_scores        = tf.stop_gradient(self.target_q_network(self.next_observation))
            tf.summary.histogram("target_action_scores", self.next_action_scores)
            # placeholder for rewards
            self.rewards                   = tf.placeholder(tf.float32, (None,), name="rewards")
            # get maximum of output from hidden layer in [1,] dimension
            target_values                  = tf.reduce_max(self.next_action_scores, reduction_indices=[1,]) * self.next_observation_mask
            # future rewards according to q value function value of rewards will be reduce in time
            self.future_rewards            = self.rewards + self.discount_rate * target_values

        with tf.name_scope("q_value_precition"):
            # FOR PREDICTION ERROR
            # TODO: What is this?
            self.action_mask                = tf.placeholder(tf.float32, (None, self.num_actions), name="action_mask")
            #  (TODO: Why action_scores * action_mask) get sum of element 
            # in tensor in [1,] dimension (sum because Q value is iterative process)
            self.masked_action_scores       = tf.reduce_sum(self.action_scores*self.action_mask, reduction_indices=[1,])
            # temporal difference in Q learning
            temp_diff                       = self.masked_action_scores - self.future_rewards
            # Get mean squared prediction error (MSPE) 
            self.prediction_error           = tf.reduce_mean(tf.square(temp_diff))
            
            gradients                       = self.optimizer.compute_gradients(self.prediction_error)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, 5), var)
            # Add histograms for gradients.
            for grad, var in gradients:
                tf.summary.histogram(var.name, var)
                if grad is not None:
                    tf.summary.histogram(var.name + '/gradients', grad)
            self.train_op                   = self.optimizer.apply_gradients(gradients)

        # UPDATE TARGET NETWORK
        with tf.name_scope("target_network_update"):
            self.target_network_update = []
            for v_source, v_target in zip(self.q_network.variables(), self.target_q_network.variables()):
                # this is equivalent to target = (1-alpha) * target + alpha * source
                update_op = v_target.assign_sub(self.target_network_update_rate * (v_target - v_source))
                self.target_network_update.append(update_op)
            self.target_network_update = tf.group(*self.target_network_update)

        # summaries
        tf.summary.scalar("prediction_error", self.prediction_error)

        self.summarize = tf.summary.merge_all()
        # no operation
        self.no_op1    = tf.no_op()

    def action(self, observation) -> int:
        """Given observation returns the action that should be chosen using
        DeepQ learning strategy. Does not backprop."""
        assert observation.shape == self.observation_shape, \
                "Action is performed based on single observation."

        self.actions_executed_so_far += 1
        exploration_p = self.linear_annealing(self.actions_executed_so_far,
                                            self.exploration_period,
                                            self.start_random_rate,
                                            self.random_action_probability)
        
        # logger.info(f'action_scores:{self.action_scores}')
        # Needs of exploration to prevent overfitting problem
        if random.random() < exploration_p:
            # rand_act = np.zeros(self.num_actions)
            # for i in range(len(rand_act)):
            #     # TODO: change random range to min and max of each parameter
            #     rand_act[i] = random.randint(0, 20)
            # rand_act = rand_act[np.newaxis,:]
            
            # Random action with k size
            rand_act = random.sample(range(self.num_actions), self.k_action)
            
            logger.info('Randomly chose action {0}'.format(rand_act))
            return rand_act
        # Run prediction to get action to tune
        else:
            # here self.s.run returns numpy.int64
            act = self.s.run(self.predicted_actions, {self.observation: observation[np.newaxis,:]})
            k = -1*abs(self.k_action)
            k_act = act.argsort()[0][k:][::-1]
            logger.info('Choose calculated action {0}'.format(k_act))
            return k_act

    def exploration_completed(self):
        return min(float(self.actions_executed_so_far) / self.exploration_period, 1.0)

    def training_step(self, samples) -> float:
        """Pick a self.minibatch_size experience from reply buffer
        and backpropage the value function.

        :return: prediction error
        """
        self.number_of_times_train_called += 1

        if self.number_of_times_train_called % self.train_every_nth == 0:
            # bach states
            states         = np.empty(self.observation_batch_shape(len(samples)))
            newstates      = np.empty(self.observation_batch_shape(len(samples)))
            action_mask    = np.zeros((len(samples),self.num_actions))     # TODO: maybe set this to 1D nparray

            newstates_mask = np.empty((len(samples),))
            rewards        = np.empty((len(samples),))

            for i, (state, action, reward, newstate, _) in enumerate(samples):
                logger.info(f'samples: {state};{action};{reward};{newstate}')
                states[i] = state
                action_mask[i] = 0
                for action_id in action:
                    action_mask[i][action_id] = 1
                # action_mask[i] = action      # TODO: action is interger here
                rewards[i] = reward
                if newstate is not None:
                    newstates[i] = newstate
                    newstates_mask[i] = 1 
                else:
                    newstates[i] = 0
                    newstates_mask[i] = 0

            calculate_summaries = self.iteration % 100 == 0 and \
                    self.summary_writer is not None

            cost, _, summary_str = self.s.run([
                self.prediction_error,
                # apply gradient with observation, next observation, action, and reward
                self.train_op,
                self.summarize if calculate_summaries else self.no_op1,
            ], {
                # All placeholder is here
                self.observation:            states,
                self.next_observation:       newstates,
                self.next_observation_mask:  newstates_mask,
                self.action_mask:            action_mask,
                self.rewards:                rewards,
            })
            
            self.s.run(self.target_network_update)

            if calculate_summaries:
                self.summary_writer.add_summary(summary_str, self.iteration)

            self.iteration += 1
            return cost
        else:
            return None

    def save(self, save_dir):
        STATE_FILE = os.path.join(save_dir, 'deepq_state')
        MODEL_FILE = os.path.join(save_dir, 'model')

        # deepq state
        state = {
            'actions_executed_so_far':      self.actions_executed_so_far,
            'iteration':                    self.iteration,
            #'number_of_times_store_called': self.number_of_times_store_called,
            'number_of_times_train_called': self.number_of_times_train_called,
        }

        print('Saving model...')

        saving_started = time.time()

        self.saver.save(self.s, MODEL_FILE)
        with open(STATE_FILE, "wb") as f:
            pickle.dump(state, f)

        print('Model saved in {} s'.format(time.time() - saving_started))

    def restore(self, save_dir):
        # deepq state
        STATE_FILE      = os.path.join(save_dir, 'deepq_state')
        MODEL_FILE      = os.path.join(save_dir, 'model')

        with open(STATE_FILE, "rb") as f:
            state = pickle.load(f)
        self.saver.restore(self.s, MODEL_FILE)

        self.actions_executed_so_far      = state['actions_executed_so_far']
        self.iteration                    = state['iteration']
        #self.number_of_times_store_called = state['number_of_times_store_called']
        self.number_of_times_train_called = state['number_of_times_train_called']



