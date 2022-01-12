# Read composition of training, test, validation folder of client_id and recreate it

import os,sys,inspect
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from Classes.Params import param
from scipy.io import savemat
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
cwd = currentdir
import numpy as np
import shutil
from distutils.dir_util import copy_tree
import difflib
import re
import pickle

exam_modalities = { 1: '\\only_flair',
                    4: '\\flair_t1_t1ce_t2'
                }

client_id = '0'

# Folder of tf_records
tf_record_folder = cwd + '\\Training_res\\tf_records'  
tf_record_dataset_folder = tf_record_folder + '\\tf_records_' + param.DATASET_COMPOSITION + exam_modalities[param.number_medical_exams]

training_record = tf_record_dataset_folder + '\\H{}\\training'.format(client_id) + '\\tfrecord'
test_record = tf_record_dataset_folder + '\\test' + '\\tfrecord'
validation_record = tf_record_dataset_folder + '\\validation' + '\\tfrecord'

# Load composition of dataset
patients_npy_file = tf_record_dataset_folder + '\\H{}'.format(client_id) + '\\patients.npy'
res = np.load(patients_npy_file, allow_pickle = True).ravel()
res = res[0]
res_train = res['training']
res_valid = res['validation']
res_test = res['test']

training_files = 0
validation_files = 0
test_files = 1

# Origin of files
brats2018_hgg = cwd + '\\..\\Brats2018\\original\\HGG'
brats2018_lgg = cwd + '\\..\\Brats2018\\original\\LGG'
athens_train = cwd + '\\..\\Atene\\training'
athens_test = cwd + '\\..\\Atene\\test'
athens_validation = cwd + '\\..\\Atene\\validation'
origin = [brats2018_hgg, brats2018_lgg, athens_train, athens_test, athens_validation]

if training_files:
    # Folder to place files
    filename = (cwd + '\\H{}\\training').format(client_id)
    # Create folder if does not exists
    if not os.path.exists(filename):
        os.makedirs(filename)
    # Remove tf_record and copy the correct one
    if os.path.isfile(filename + '\\tfrecord'):
        os.remove(filename + '\\tfrecord')  
    # shutil.copyfile(training_record, filename + '\\tfrecord')
    # Remove all folders of patiens
    patients = [next(os.walk(filename))[0] + '\\' + s for s in next(os.walk(filename))[1]]
    for patient in patients:
        if os.path.exists(patient):
            shutil.rmtree(patient)
    # Search for the original file and copy it
    for patient in res_train:
        patient_name = patient.split('\\')[-1]
        found = 0
        for or_ in origin:
            # All patients in or
            origin_name = next(os.walk(or_))[1]
            for patient_name_origin in origin_name:
                if patient_name_origin == patient_name:
                    found = 1
                    # Copy folder
                    copy_tree(or_ + '\\' + patient_name, filename + '\\' + patient_name)
                    break
            if found:
                break


if validation_files:
    # Folder to place files
    filename = (cwd + '\\H{}\\validation').format(client_id)
    # Create folder if does not exists
    if not os.path.exists(filename):
        os.makedirs(filename)
    # Remove tf_record and copy the correct one
    if os.path.isfile(filename + '\\tfrecord'):
        os.remove(filename + '\\tfrecord')  
    # shutil.copyfile(validation_record, filename + '\\tfrecord')
    # Remove all folders of patiens
    patients = [next(os.walk(filename))[0] + '\\' + s for s in next(os.walk(filename))[1]]
    for patient in patients:
        if os.path.exists(patient):
            shutil.rmtree(patient)
    # Search for the original file and copy it
    for patient in res_valid:
        patient_name = patient.split('\\')[-1]
        found = 0
        for or_ in origin:
            # All patients in or
            origin_name = next(os.walk(or_))[1]
            for patient_name_origin in origin_name:
                if patient_name_origin == patient_name:
                    found = 1
                    # Copy folder
                    copy_tree(or_ + '\\' + patient_name, filename + '\\' + patient_name)
                    break
            if found:
                break

if test_files:
    # Folder to place files
    filename = (cwd + '\\H{}\\test').format(client_id)
    # Create folder if does not exists
    if not os.path.exists(filename):
        os.makedirs(filename)
    # Remove tf_record and copy the correct one
    if os.path.isfile(filename + '\\tfrecord'):
        os.remove(filename + '\\tfrecord')  
    # shutil.copyfile(test_record, filename + '\\tfrecord')
    # Remove all folders of patiens
    patients = [next(os.walk(filename))[0] + '\\' + s for s in next(os.walk(filename))[1]]
    for patient in patients:
        if os.path.exists(patient):
            shutil.rmtree(patient)
    # Search for the original file and copy it
    for patient in res_test:
        patient_name = patient.split('\\')[-1]
        found = 0
        for or_ in origin:
            # All patients in or
            origin_name = next(os.walk(or_))[1]
            for patient_name_origin in origin_name:
                if patient_name_origin == patient_name:
                    found = 1
                    # Copy folder
                    copy_tree(or_ + '\\' + patient_name, filename + '\\' + patient_name)
                    break
            if found:
                break