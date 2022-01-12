import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
import sys
from scipy.ndimage.interpolation import shift
import SimpleITK as sitk
from glob import glob
import random
from dltk.io.augmentation import *
from dltk.io.preprocessing import *
from pathlib import Path
import matplotlib.pyplot as plt
from pydicom import dcmread
from pydicom.data import get_testdata_file
import re
import difflib
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
cwd = currentdir
print(cwd)


# start = os.path.join('Training_res', 'tf_records', 'tf_records_LGG', 'flair_t1_t1ce_t2')
# dir1 = os.path.join(cwd, 'H0', 'training', 'tfrecord')
# dir2 = os.path.join(cwd, 'H1', 'training', 'tfrecord')
# dir3 = os.path.join(cwd, 'H2', 'training', 'tfrecord')


dir1 = os.path.join(cwd, 'H3', 'training', 'tfrecord')
dir2 = os.path.join(cwd, 'H4', 'training', 'tfrecord')


# dest = os.path.join(cwd, start, 'centralized', 'training', 'tfrecord')


dest = os.path.join(cwd, 'H4', 'tfrecord')

# Create dataset from multiple .tfrecord files
list_of_tfrecord_files = [dir1, dir2]
dataset = tf.data.TFRecordDataset(list_of_tfrecord_files)

# Save dataset to .tfrecord file
writer = tf.data.experimental.TFRecordWriter(dest)
writer.write(dataset)