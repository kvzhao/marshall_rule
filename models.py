import numpy as np
import tensorflow as tf

def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar(name+'_mean', mean)
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
    def __init__ (self, hidden_sizes):
        """
            hidden_size: list of size of hidden layers
        """
        self.name = 'network'
        self.hidden_sizes = hidden_sizes
    
    def build_network(self, x):
        with tf.variable_scope(self.name):
            h = tf.nn.relu(linear(x, self.hidden_sizes[0], "input", normalized_columns_initializer(0.01)))
            print ("[MODEL] input layer with size {}".format(self.hidden_sizes[0]))
            variable_summaries(h, "input")
            for i, size in enumerate(self.hidden_sizes[1:]):
                print ("[MODEL] layer {} with size {}".format(i+1, size))
                h = tf.nn.relu(linear(h, size, "hidden{}".format(i+1), normalized_columns_initializer(0.01)))
                variable_summaries(h, "hidden{}".format(i+1))
            h = tf.nn.relu(linear(h, 2, "output", normalized_columns_initializer(0.01)))
            print ("[MODEL] out layer with size {}".format(2))
            variable_summaries(h, "output")
        return h

    def __call__ (self, x):
        return self.build_network(x)

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
