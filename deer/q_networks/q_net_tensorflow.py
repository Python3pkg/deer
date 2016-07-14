"""
Code for general deep Q-learning using tensorflow that can take as inputs scalars, vectors and matrices

.. Author: Vincent Francois-Lavet
"""

import numpy as np
import tensorflow as tf
from ..base_classes import QNetwork
from .NN_tensorflow import NN # Default Neural network used
import copy

class MyQNetwork(QNetwork):
    """
    Deep Q-learning network using tensorflow
    
    Parameters
    -----------
    environment : object from class Environment
    rho : float
        Parameter for rmsprop. Default : 0.9
    rms_epsilon : float
        Parameter for rmsprop. Default : 0.0001
    momentum : float
        Default : 0
    clip_delta : float
        Not implemented.
    freeze_interval : int
        Period during which the target network is freezed and after which the target network is updated. Default : 1000
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent. Default : 32
    network_type : str
        Not used. Default : None
    update_rule: str
        {sgd,rmsprop}. Default : rmsprop
    batch_accumulator : str
        {sum,mean}. Default : sum
    random_state : numpy random number generator
    double_Q : bool, optional
        Activate or not the double_Q learning.
        More informations in : Hado van Hasselt et al. (2015) - Deep Reinforcement Learning with Double Q-learning.
    neural_network : object, optional
        default is deer.qnetworks.NN_tensorflow
    """

    def __init__(self, environment, rho=0.9, rms_epsilon=0.0001, momentum=None, clip_delta=0, freeze_interval=1000, batch_size=32, network_type=None, update_rule="rmsprop", batch_accumulator="sum", random_state=np.random.RandomState(), double_Q=False, neural_network=NN):
        """ Initialize environment
        
        """
        QNetwork.__init__(self,environment, batch_size)
        
        self._rho = rho
        self._rms_epsilon = rms_epsilon
        self._momentum = momentum
        self._clip_delta = clip_delta
        self._freeze_interval = freeze_interval
        self._double_Q = double_Q
        self._random_state = random_state
        self.update_counter = 0
        
        self.states=[]   # list of symbolic variables for each of the k element in the belief state
                    # --> [ T.tensor4 if observation of element=matrix, T.tensor3 if vector, T.tensor 2 if scalar ]
        self.next_states=[] # idem than states at t+1 
        self.states_shared=[] # list of shared variable for each of the k element in the belief state
        self.next_states_shared=[] # idem that self.states_shared at t+1

        for i, dim in enumerate(self._input_dimensions):
            if len(dim) == 3:
                self.states.append( tf.placeholder(tf.float32, [None, dim[1], dim[2], dim[0]]) ) #BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS
                self.next_states.append( tf.placeholder(tf.float32, [None, dim[0], dim[1], dim[2]]) )

            elif len(dim) == 2:
                self.states.append( tf.placeholder(tf.float32, [None, dim[0], dim[1], 1]) )
                vnext_states.append( tf.placeholder(tf.float32, [None, dim[0], dim[1], 1]) )
                
            elif len(dim) == 1:            
                self.states.append( tf.placeholder(tf.float32, [None, dim[0]]) )
                self.next_states.append( tf.placeholder(tf.float32, [None, dim[0]]) )
                        
        print("Number of observations per state: {}".format(len(self._input_dimensions)))
        print("For each observation, historySize + ponctualObs_i.shape: {}".format(self._input_dimensions))

        QNet=neural_network(self._batch_size, self._input_dimensions, self._n_actions, self._random_state)
        self.q_vals, self.params, shape_after_conv = QNet._buildDQN(self.states)#, self.params, shape_after_conv = QNet._buildDQN(states)

        print("Number of neurons after spatial and temporal convolution layers: {}".format(shape_after_conv))
        
        self.next_q_vals, self.next_params, shape_after_conv = QNet._buildDQN(self.next_states)
        
        self.target_q_vals=tf.placeholder(tf.float32, self.next_q_vals.get_shape())
        
        self.loss = tf.reduce_sum(tf.square(self.q_vals - self.target_q_vals))/batch_size #tf.reduce_mean(tf.square(self.q_vals - self.target_q_vals)) #loss = tf.nn.l2_loss(diff)
        
        if update_rule == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(self._lr, self._rho, self._momentum, self._rms_epsilon)

        elif update_rule == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self._lr)
        else:
            raise ValueError("Unrecognized update: {}".format(update_rule))
    

        
        self.performGD = optimizer.minimize(self.loss)

        #init_op = tf.initialize_all_variables()
        tf.set_random_seed(random_state.randint(1024))
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        self._resetQHat()        
        
#    def getAllParams(self):
#        #FIXME
#        
#    def setAllParams(self, list_of_values):
#        #FIXME
        
    def train(self, states_val, actions_val, rewards_val, next_states_val, terminals_val):
        """
        Train one batch.

        1. Set shared variable in states_shared, next_states_shared, actions_shared, rewards_shared, terminals_shared         
        2. perform batch training

        Parameters
        -----------
        states_val : list of batch_size * [list of max_num_elements* [list of k * [element 2D,1D or scalar]])
        actions_val : b x 1 numpy array of integers
        rewards_val : b x 1 numpy array
        next_states_val : list of batch_size * [list of max_num_elements* [list of k * [element 2D,1D or scalar]])
        terminals_val : b x 1 numpy boolean array (currently ignored)


        Returns
        -------
        average loss of the batch training
        """
        
        if self.update_counter % self._freeze_interval == 0: #FIXME
            self._resetQHat()
        
        next_q_vals = self.sess.run(self.next_q_vals,feed_dict={self.next_states[0]:next_states_val[0], self.next_states[1]:next_states_val[1]})
        
        max_next_q_vals=np.max(next_q_vals, axis=1, keepdims=True)

        not_terminals=np.ones_like(terminals_val) - terminals_val
        
        target = rewards_val + not_terminals * self._df * max_next_q_vals.reshape((-1))
                
        q_vals = self.sess.run(self.q_vals,feed_dict={self.states[0]:states_val[0], self.states[1]:states_val[1]})

        q_val = q_vals[  np.arange(self._batch_size), actions_val.reshape((-1,))  ]

        diff = - q_val + target

        q_vals_copy=np.array(copy.deepcopy(q_vals)).astype(np.float32)   
        
        q_vals_copy[  np.arange(self._batch_size), actions_val.reshape((-1,))  ] = target

        self.sess.run(self.performGD,feed_dict={self.target_q_vals:q_vals_copy, self.states[0]:states_val[0], self.states[1]:states_val[1]})
        
        self.update_counter += 1
        
        loss_ind=np.square(diff)

        return np.average(loss_ind),loss_ind


#        # Full tensorflow impossible?
#        max_next_q_vals=tf.reduce_max(self.next_q_vals, reduction_indices=1, keep_dims=True)
#        not_terminals=tf.ones(shape=tf.shape(terminals)) - terminals
#        target = rewards + not_terminals * thediscount * max_next_q_vals
#
#        # Impossible to do with tensorflow ?? https://github.com/tensorflow/tensorflow/issues/418
#        print self.q_vals[ tf.constant(np.arange(batch_size)), tf.reshape(actions,[32])]
#        
#        q_val=tf.reshape( self.q_vals[ tf.constant(np.arange(batch_size)), tf.reshape(actions,[32])] , [-1, 1] )
#        diff = - q_val + target

    def qValues(self, state_val):
        """ Get the q values for a belief state

        Arguments
        ---------
        state_val : one belief state

        Returns
        -------
        The q value for the provided belief state
        """ 
        
        return self.sess.run(self.q_vals,feed_dict={self.states[0]:np.expand_dims(state_val[0],axis=0), self.states[1]:np.expand_dims(state_val[1],axis=0)})[0]

    def chooseBestAction(self, state):
        """ Get the best action for a belief state

        Arguments
        ---------
        state : one belief state

        Returns
        -------
        The best action : int
        """        
        q_vals = self.qValues(state)

        return np.argmax(q_vals)
        
    def _resetQHat(self):
        for i,(param,next_param) in enumerate(zip(self.params, self.next_params)):
            self.sess.run( next_param.assign(param) )

