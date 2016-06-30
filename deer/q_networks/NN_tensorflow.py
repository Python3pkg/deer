"""
Neural network using Tensorflow (called by q_net_tensorflow)

.. Author: Vincent Francois-Lavet
"""

import numpy as np
import tensorflow as tf
import math

class NN():
    """
    Deep Q-learning network using Theano
    
    Parameters
    -----------
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent
    input_dimensions :
    n_actions :
    random_state : numpy random number generator
    """
    def __init__(self, batch_size, input_dimensions, n_actions, random_state):
        self._input_dimensions=input_dimensions
        self._batch_size=batch_size
        self._random_state=random_state
        self._n_actions=n_actions
                
    def _buildDQN(self, inputs):
        """
        Build a network consistent with each type of inputs
        """
        layers=[]
        outs_conv=[]
        outs_conv_shapes=[]
        
        for i, dim in enumerate(self._input_dimensions):
            nfilter=[]
            
            # - observation[i] is a FRAME -
            if len(dim) == 3: 

                outs_conv.append(l_conv3.output)
                outs_conv_shapes.append((nfilter[2],newR,newC))

                
            # - observation[i] is a VECTOR -
            elif len(dim) == 2 and dim[0] > 3:                
                
                outs_conv.append(l_conv2.output)
                outs_conv_shapes.append((nfilter[1],newR,newC))


            # - observation[i] is a SCALAR -
            else:
                if dim[0] > 3:

                    newR = 1
                    newC = dim[0]
                    
                    fR=1  # filter Rows
                    fC=2  # filter Column
                    pR=1  # pool Rows
                    pC=1  # pool Column
                    nfilter.append(8)
                    stride_size=1

                    conv1_weights = tf.Variable(
                        tf.truncated_normal([fR, fC, 1, nfilter[0]],
                                            stddev=0.1,
                                            seed=self._random_state.randint(-1024,1024) ))
                    conv1_biases = tf.Variable(tf.zeros([nfilter[0]]))

                    conv = tf.nn.conv2d(tf.reshape(inputs[i],[-1,1,dim[0],1]),
                                        filter=conv1_weights,
                                        strides=[stride_size, stride_size, 1, 1],
                                        padding='VALID')

                    # Bias and rectified linear non-linearity.
                    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
                    
                    layers.append(conv)
                    layers.append(relu)
                    
                    
                    newC = (newC - fC + 1 - pC) // stride_size + 1  # stride 2

                    fR=1  # filter Rows
                    fC=2  # filter Column
                    pR=1  # pool Rows
                    pC=1  # pool Column
                    nfilter.append(8)
                    stride_size=1
                    
                    conv2_weights = tf.Variable(
                        tf.truncated_normal([fR, fC, nfilter[0], nfilter[1]],  # 5x5 filter, depth 32.
                                            stddev=0.1,
                                            seed=self._random_state.randint(-1024,1024) ))
                    conv2_biases = tf.Variable(tf.zeros([nfilter[1]]))

                    conv = tf.nn.conv2d(relu,
                                        filter=conv2_weights,
                                        strides=[stride_size, stride_size, 1, 1],
                                        padding='VALID')

                    # Bias and rectified linear non-linearity.
                    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
                    
                    layers.append(conv)
                    layers.append(relu)

                    
                    newC = (newC - fC + 1 - pC) // stride_size + 1  # stride 2
                    print nfilter[1],newC
                    outs_conv_shapes.append((nfilter[1],newC))
                    outs_conv.append(conv)
                    
                else:
                    if(len(dim) == 2):
                        outs_conv_shapes.append((dim[0],dim[1]))
                    elif(len(dim) == 1):
                        outs_conv_shapes.append((1,dim[0]))
                    outs_conv.append(inputs[i])
        
        
        ## Custom merge of layers
        print outs_conv_shapes
        output_conv = tf.reshape( outs_conv[0] , (self._batch_size, np.prod(outs_conv_shapes[0])) )
        shapes=np.prod(outs_conv_shapes[0])

        if (len(outs_conv)>1):
            for out_conv,out_conv_shape in zip(outs_conv[1:],outs_conv_shapes[1:]):
                output_conv=tf.concat(1, (output_conv, tf.reshape( out_conv, (self._batch_size, np.prod(out_conv_shape)) ) ))#, axis=1))
                shapes+=np.prod(out_conv_shape)
                shapes
                
        # Hidden 1
        hidden1_units=50
        with tf.name_scope('hidden1'):
            weights = tf.Variable(
                tf.truncated_normal([shapes, hidden1_units],
                                    stddev=1.0 / math.sqrt(float(shapes))),
                name='weights')
            biases = tf.Variable(tf.zeros([hidden1_units]),
                                 name='biases')
            hidden1 = tf.nn.relu(tf.matmul(output_conv, weights) + biases)
        layers.append(hidden1)

        # Hidden 2
        hidden2_units=20
        with tf.name_scope('hidden2'):
            weights = tf.Variable(
                tf.truncated_normal([hidden1_units, hidden2_units],
                                    stddev=1.0 / math.sqrt(float(hidden1_units))),
                name='weights')
            biases = tf.Variable(tf.zeros([hidden2_units]),
                                 name='biases')
            hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
        layers.append(hidden2)


        # outLayer
        with tf.name_scope('out'):
            weights = tf.Variable(
                tf.truncated_normal([hidden2_units, self._n_actions],
                                    stddev=1.0 / math.sqrt(float(hidden2_units))),
                name='weights')
            biases = tf.Variable(tf.zeros([self._n_actions]),
                                 name='biases')
            out = tf.nn.relu(tf.matmul(hidden2, weights) + biases)
        layers.append(out)

        
        return out#.output#, params, outs_conv_shapes

if __name__ == '__main__':
    pass
