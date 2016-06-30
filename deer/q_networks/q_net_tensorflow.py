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
    Deep Q-learning network using Theano
    
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
        default is deer.qnetworks.NN_keras
    """

    def __init__(self, environment, rho=0.9, rms_epsilon=0.0001, momentum=None, clip_delta=0, freeze_interval=1000, batch_size=32, network_type=None, update_rule="rmsprop", batch_accumulator="sum", random_state=np.random.RandomState(), double_Q=False, neural_network=NN):
        """ Initialize environment
        
        """
        QNetwork.__init__(self,environment, batch_size)

        
        self.rho = rho
        self.rms_epsilon = rms_epsilon
        self.momentum = momentum
        self.clip_delta = clip_delta
        self.freeze_interval = freeze_interval
        self._double_Q = double_Q
        self._random_state = random_state
        self.update_counter = 0
        
        self.states=[]   # list of symbolic variables for each of the k element in the belief state
                    # --> [ T.tensor4 if observation of element=matrix, T.tensor3 if vector, T.tensor 2 if scalar ]
        self.next_states=[] # idem than states at t+1 
        self.states_shared=[] # list of shared variable for each of the k element in the belief state
        self.next_states_shared=[] # idem that self.states_shared at t+1

        for i, dim in enumerate(self._input_dimensions):
            print dim
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
        self.q_vals = QNet._buildDQN(self.states)#, self.params, shape_after_conv = QNet._buildDQN(states)
        
        self.next_q_vals = QNet._buildDQN(self.next_states)
        
        #self._resetQHat() # FIXME        
        
        #init_op = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        
    def toDump(self):
        # FIXME

        return None,None

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
        
        #if self.update_counter % self._freeze_interval == 0: #FIXME
        #    self._resetQHat()
        
        #feed_dict = {x: [your_image]}
#        with tf.Session() as sess:
#            tf.initialize_all_variables().run()
        next_q_vals = self.sess.run(self.next_q_vals,feed_dict={self.next_states[0]:next_states_val[0], self.next_states[1]:next_states_val[1]})
        
        print "next_q_vals"
        print next_q_vals
        
        #FIXME        
#        if(self._double_Q==True):
#        else:

        max_next_q_vals=np.max(next_q_vals, axis=1, keepdims=True)

        not_terminals=np.ones_like(terminals_val) - terminals_val
        
        target = rewards_val + not_terminals * self._df * max_next_q_vals.reshape((-1))
                
        q_vals = self.sess.run(self.q_vals,feed_dict={self.states[0]:states_val[0], self.states[1]:states_val[1]})

        print "q_vals before"
        print q_vals

        q_val = q_vals[  np.arange(self._batch_size), actions_val.reshape((-1,))  ]

        print q_val
        diff = - q_val + target
        print "diff"       
        print diff        

        q_vals_copy=copy(q_vals)
        q_vals[  np.arange(self._batch_size), actions_val.reshape((-1,))  ] = target
        print "q_vals after"
        print q_vals
        print q_vals_copy

#        # Minimize the squared errors.
#        loss = tf.nn.l2_loss(diff) #tf.reduce_mean(tf.square(self.next_q_vals - q_vals))
#        optimizer = tf.train.GradientDescentOptimizer(0.5)
#        train = optimizer.minimize(loss)
        
        print loss
        
        self.update_counter += 1


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