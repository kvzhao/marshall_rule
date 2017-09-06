import sys, os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Merge J2/J1 and Configurations. Generate Data & Labels')
parser.add_argument('--Jupper', dest='Jupper', default=0.0, type=int, help='Upper bound of J2/J1 value')
parser.add_argument('--Jlower', dest='Jlower', default=0.0, type=int, help='Lower bound of J2/J1 value')
parser.add_argument('--out', dest='out_postfix', default='J0', type=str, help='Postfix of output filename')

FLAGS = parser.parse_args()

files = os.listdir('.')
files.remove('merger.py')

STATES = []
SIGNS = []

Jupper = FLAGS.Jupper
Jlower = FLAGS.Jlower
out_postfix = FLAGS.out_postfix

for f in files:
    if str.startswith(f, 'sign'):
        fname = f.rstrip('.txt')
        j2j1 = float(fname.split('_')[-1])

        #
        if (j2j1 > Jupper or j2j1 < Jlower):
            print ('Ignore J = {}'.format(j2j1))
            continue
        else:
            print ('file name: {}'.format(fname))

        signs = np.loadtxt(f, dtype=np.float)
        states = np.loadtxt('states.txt')
        j2j1line = np.repeat(j2j1, states.shape[0])

        newstates = np.column_stack([j2j1line, states])
        #signs = np.loadtxt('sign_'+str(j2j1), dtype=np.float)

        STATES.extend(newstates)
        SIGNS.extend(signs)

np.savetxt('../datasetMerged/states_{}.txt'.format(out_postfix), STATES)
np.savetxt('../datasetMerged/sign_{}.txt'.format(out_postfix), SIGNS)

print ('done.')
