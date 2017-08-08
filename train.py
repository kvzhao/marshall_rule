import os
import tensorflow as tf
from data_sampler import DataSampler
from binary_classification import BinaryClassifier

tf.flags.DEFINE_boolean("sanity_check", False, "If set, training will perform on small piece of data.")
FLAGS = tf.flags.FLAGS

LOSS_FUNCS = ['xentropy', 'hinge']
SOLVERS = ['sgd', 'adam']
LEARNING_RATES = [0.001, 0.01, 0.05, 0.1]
HIDDEN_SIZES = [[32, 16, 2],
                [16, 2]]

def main():
    for solver in SOLVERS:
        for loss_type in LOSS_FUNCS:
            for h in HIDDEN_SIZES:
                print ('Hidden sizes = {}'.format(h))
                hidden_sizes = h
                for lr in LEARNING_RATES:
                    ds = DataSampler()
                    arch = 'arch' + 'x'.join([str(i) for i in hidden_sizes]) + 'lr={}'.format(lr)
                    task = '_'.join([arch, loss_type, solver])
                    print ('[TRAIN] Start experiment: {}'.format(task))
                    classifier = BinaryClassifier(data_sampler=ds, 
                                                task_name = task,
                                                hidden_sizes=hidden_sizes,
                                                solver_type=solver,
                                                activation='relu',
                                                loss_func=loss_type,
                                                learning_rate=lr)
                    if (FLAGS.sanity_check):
                        classifier.overfit_test()
                    else:
                        classifier.train()
                    # reset the model
                    tf.reset_default_graph()
                    print ('[TRAIN] Done {}, reset network.'.format(task))

if __name__ == '__main__':
    main()