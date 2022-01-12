# Save composition of training, test, validation folder of client_id

import os,sys,inspect
from scipy.io import savemat
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
cwd = currentdir
import numpy as np

import re
import pickle

client_id = '3'
res_train = []
res_valid = []
res_test = []


filename = (cwd + '\\H{}\\training').format(client_id)
patients = [next(os.walk(filename))[0] + '\\' + s for s in next(os.walk(filename))[1]]
for patient in patients:
    res_train.append(patient.split('\\')[-1])

filename = (cwd + '\\H{}\\validation').format(client_id)
patients = [next(os.walk(filename))[0] + '\\' + s for s in next(os.walk(filename))[1]]
for patient in patients:
    res_valid.append(patient.split('\\')[-1])

filename = (cwd + '\\H{}\\test').format(client_id)
patients = [next(os.walk(filename))[0] + '\\' + s for s in next(os.walk(filename))[1]]
for patient in patients:
    res_test.append(patient.split('\\')[-1])

res =  {'training':res_train, 'validation': res_valid, 'test':res_test}
np.save(cwd + '/patients.npy', res, allow_pickle = True)
print(res)