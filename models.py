import numpy as np
import tensorflow as tf

def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        tf.summary.scalar(name+'_mean', tf.reduce_mean(var))
        tf.summary.scalar(name+'_sparsity', tf.nn.zero_fraction(var))
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar(name+'_stddev', stddev)
    tf.summary.scalar(name+'_max', tf.reduce_max(var))
    tf.summary.scalar(name+'_min', tf.reduce_min(var))
    tf.summary.histogram(name+'_histogram', var)

def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class simple_network():
    def __init__ (self, hidden_size):
        self.name = 'network'
        self.hidden_size = hidden_size
    
    def __call__ (self, x):
        with tf.variable_scope(self.name):
            h1 = tf.nn.relu(linear(x, self.hidden_size, "hidden1", normalized_columns_initializer(0.01)))
            variable_summaries(h1)
            # TODO: Avoid hard-coded output size
            out = linear(h1, 2, "out")
            variable_summaries(h1, "hidden1")
            variable_summaries(out, "out_layer")
            return out

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
