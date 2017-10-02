import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.layers as layers

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

class RNN(object):
    def __init__ (self, x, x_len, cell_size, 
                    num_layers, num_classes,
                    write_summary=True, 
                    name='rnn_net',
                    use_cos=True,
                    reuse=False):
        self.name = name
        self.cell_size = cell_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.x = x
        self.x_dim = x.shape[-1]
        self.x_len = x_len
        self.reuse = reuse
        self.use_cos = use_cos
        self.write_summary = write_summary

        self._build()

    def _build(self):
        with tf.variable_scope(self.name, reuse=self.reuse):
            print ('Building RNN network')

            if self.use_cos:
                # Resize and so on
                self.x = tf.squeeze(self.x)
                cos_w = tf.get_variable("cos_weight", shape=[self.x_dim, self.cell_size], dtype=tf.float32)
                cos_b = tf.get_variable("cos_bias", shape=[self.cell_size], dtype=tf.float32)
                x = tf.cos( tf.add(tf.matmul(self.x, cos_w), cos_b))
                x = tf.expand_dims(x, axis=1)
            else:
                x = self.x

            if self.num_layers == 1:
                lstm_cell = rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, 
                                            reuse=tf.get_variable_scope().reuse)
            else:
                lstm_cell = []
                for _ in range(self.num_layers):
                    cell = rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, 
                                            reuse=tf.get_variable_scope().reuse)
                    lstm_cell.append(cell)
                lstm_cell = rnn.MultiRNNCell(lstm_cell)

            outputs, states = tf.nn.dynamic_rnn(lstm_cell, 
                                inputs=x,
                                sequence_length=self.x_len,
                                dtype=tf.float32,
                                time_major=False)
            
            # transpose to time-major, then get time dim
            outputs = tf.transpose(outputs, [1, 0, 2])
            outshape = outputs[-1].get_shape()
            print ('outshape = outputs[-1] = {}, this number should be connected to projection layer'.format(outshape))

            w = tf.get_variable("w", [outshape[-1], self.num_classes], 
                                initializer=tf.random_normal_initializer(),
                                regularizer=layers.l2_regularizer(1.0))
            b = tf.get_variable("b", [self.num_classes], 
                                initializer=tf.constant_initializer(0.001),
                                regularizer=layers.l2_regularizer(1.0))
            linout = tf.matmul(outputs[-1], w) + b
            if (self.write_summary):
                variable_summaries(w, self.name + '_w')
                variable_summaries(b, self.name + '_b')
            self.logits = linout
            self.lstm_states = states

    def __call__(self):
        return self.logits, self.lstm_states

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
