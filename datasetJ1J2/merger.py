import sys, os
import numpy as np

files = os.listdir('.')
files.remove('merger.py')

STATES = []
SIGNS = []

for f in files:
    if str.startswith(f, 'states'):
        j2j1 = float(f.split('_')[-1])
        states = np.loadtxt(f, dtype=np.float)
        j2j1line = np.repeat(j2j1, states.shape[0])
        newstates = np.column_stack([j2j1line, states])
        signs = np.loadtxt('sign_'+str(j2j1), dtype=np.float)

        STATES.extend(newstates)
        SIGNS.extend(signs)
np.savetxt('states.txt', STATES)
np.savetxt('sign.txt', SIGNS)
