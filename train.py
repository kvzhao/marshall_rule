import os
import tensorflow as tf
from data_sampler import DataSampler
from binary_classification import BinaryClassifier
from constants import MODEL

tf.flags.DEFINE_boolean("sanity_check", False, "If set, training will perform on small piece of data.")
FLAGS = tf.flags.FLAGS

LOSS_FUNCS = ['xentropy', 'hinge']
SOLVERS = ['adam', 'sgd']
LEARNING_RATES = [0.001, 0.005, 0.01, 0.05]
HIDDEN_SIZES = [[8, 2], 
                [16, 2],
                [32, 2],
                [64, 2],
                [128, 2],
                [256, 2]]

LOSS_FUNCS = ['xentropy']
SOLVERS = ['adam']
LEARNING_RATES = [0.001]
HIDDEN_SIZES = [[8, 2]]

def main():
    for solver in SOLVERS:
        for loss_type in LOSS_FUNCS:
            for h in HIDDEN_SIZES:
                print ('Hidden sizes = {}'.format(h))
                hidden_sizes = h
                for lr in LEARNING_RATES:
                    ds = DataSampler()
                    arch = MODEL + '_' + 'x'.join([str(i) for i in hidden_sizes]) + 'x{}'.format(lr)
                    task = '_'.join([arch, loss_type, solver])
                    print ('[TRAIN] Start experiment: {}'.format(task))
                    classifier = BinaryClassifier(data_sampler=ds,
                                                task_name = task,
                                                hidden_sizes=hidden_sizes,
                                                model = MODEL, 
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