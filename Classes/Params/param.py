import os
import platform
OS = platform.system()
if OS == 'Darwin':
    OS = 'MACOS'
    import appscript
elif OS == 'Windows':
    OS = 'WINDOWS'
elif OS == 'Linux':
    OS = 'UBUNTU'
print('\n' + OS)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if OS != 'WINDOWS':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # -1 to not use GPU

import tensorflow as tf
if OS == 'UBUNTU' or OS == 'WINDOWS':

    try:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        if OS == 'WINDOWS':
            tf.config.experimental.set_virtual_device_configuration( physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096*1)])
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
from scipy.ndimage.interpolation import shift
import multiprocessing
import copy
import time
import traceback
import sys
import re
import importlib
import subprocess
import psutil
import random
import dill
import zlib
import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
import paho.mqtt.subscribe as subscribe
import socket
if OS == 'MACOS':
    import appscript
import struct
import inspect
from pathlib import Path
from math import *
import _pickle as cPickle
import json
from Classes.Optimizers.pgd import PerturbedGradientDescent
from Classes.Optimizers.AMSGrad import AMSGrad
from Classes.Optimizers.Adam import Adam
from Classes.Metrics.evaluation_metrics import *
from Classes.Losses.losses import *

#########################FIRST
# Settings
REMOTE = 1                                                 # Test local or remote
TLS_USAGE = 1
LAUNCH_MQTT_BROKER = 0

# Seed for reproducibility
SEED = 1234 # 1234
tf.random.set_seed(SEED) 
np.random.seed(SEED) 
cwd = os.path.split(os.path.abspath(__file__))[0]  # os.getcwd()           # os.path.split(os.path.abspath(__file__))[0]
cwd = str(Path(Path(cwd).parent.absolute()).parent.absolute())
save_weights = True                                                # Save weights of each federated round

# Directories for WINDOWS
python_conda_dir = sys.executable 
MQTT_broker_dir = '"F:\\bernardo_camajori_tedechini\Mosquitto\mosquitto.exe"'        
MQTT_broker_config_file = '"F:\\bernardo_camajori_tedechini\Mosquitto\mosquitto.conf"'   

# Continual learning
CONTINUAL_LEARNING = 0
INITIAL_ROUND = -1  # -1 50
MODEL_WEIGHTS_FILE = os.path.join(cwd, 'H0', 'saved_weights', '{}'.format(INITIAL_ROUND), 'weights.npy')

# Environment
try:
    conda_env = os.environ['CONDA_DEFAULT_ENV'] 
except:
    conda_env = sys.executable.split(os.sep)[3]

# MQTT Topics and Broker address
localhost ='127.0.0.1'  # '128.141.183.190'
with open(os.path.join(cwd, 'MQTT_broker_config.json')) as f:
    config = json.load(f)

if not REMOTE:
    broker_address = localhost
else:
    broker_address = config['broker_address']

if not TLS_USAGE:
    MQTT_port = 1885   
    TLS_ = None       
    AUTH_ = None   
else:
    MQTT_port = config['MQTT_port']   # 11883(linux cern) 
    TLS_ = config['TLS_']       
    TLS_['tls_version'] = mqtt.ssl.PROTOCOL_TLS
    AUTH_ = config['AUTH_']    

server_weights_topic = config["Name_of_federation"] + 'server_weights'
client_weights_topic = config["Name_of_federation"] + 'client_weights'
params_topic =  config["Name_of_federation"] + 'params'
samples_topic = config["Name_of_federation"] + 'samples'
Last_seen_time_topic = config["Name_of_federation"] + 'Last_seen_time' 


#########################SECOND
TYPES_DATASET = {
    0: ['Mnist', 'Classes.Datasets.MNIST', 'MNIST'],
    1: ['Mnist_non_iid', 'Classes.Datasets.MNIST_non_iid_label', 'MNIST'],
    2: ['Mnist_non_iid', 'Classes.Datasets.MNIST_non_iid_noise', 'MNIST'],
    3: ['Brats', 'Classes.Datasets.Brats', 'BRATS']
}
TYPES_MODELS = {
    0: ['Classes.Models.CNN', 'CNN'],
    1: ['Classes.Models.CNN_BN', 'CNN_BN'],
    2: ['Classes.Models.ResNet50', 'RESNET50'],
    3: ['Classes.Models.ResNet', 'RESNET'],
    4: ['Classes.Models.ResNetPreAct', 'RESNETPREACT'],
    5: ['Classes.Models.Unet_2D', 'UNET_2D'],               # B = 16 with 4GByte for Brats / 64 with 24GByte for Athens (32 filters) / 32 with 24GByte for Brats.    with 4GByte for Brats / 32 with 24GByte for Athens (64 filters)
    6: ['Classes.Models.Unet_2D_cust', 'UNET_2D_cust'],     # B = 2  with 4GByte for Brats / 16 with 24GByte for Athens
    7: ['Classes.Models.Unet_2D_dense', 'UNET_2D_dense'],    # B = 8  with 4GByte for Brats / 64 with 24GByte for Athens
    8: ['Classes.Models.Unet_2D_cust2', 'UNET_2D_cust2']
}  
TYPES_ALGORITHMS = {
    0: ['FedAvg', 'Classes.Fl_Algorithms.FedAvg', 'FEDAVG'],
    1: ['FedBn', 'Classes.Fl_Algorithms.FedBn', 'FEDBN'],
    2: ['FedAdp', 'Classes.Fl_Algorithms.FedAdp', 'FEDADP'],
    3: ['FedProx', 'Classes.Fl_Algorithms.FedProx', 'FEDPROX'],
    4: ['FedAdpProx', 'Classes.Fl_Algorithms.FedAdpProx', 'FEDADPPROX'],
    5: ['FedAdpProx', 'Classes.Fl_Algorithms.FedNew', 'FEDADPPROX']
}
TYPES_ARCHITECTURE = {
    0: ['PS_Synch'],            # PS-S/C-S
    1: ['PS_Asynch_PS_Synch'],  # PS-S/C-A
    2: ['PS_Asynch'],           # PS-A/C-A
    3: ['Consensus'],
}
                         
DATASET = TYPES_DATASET[3]                                      # choose
MODEL = TYPES_MODELS[5]                                         # choose
ALGORITHM = TYPES_ALGORITHMS[0]                                  # choose
ARCHITECTURE = TYPES_ARCHITECTURE[2]                             # choose

MODULE_DATASET_NAME = DATASET[1]
DATASET_NAME = DATASET[2]
MODULE_MODEL_NAME = MODEL[0]
MODEL_NAME = MODEL[1]
MODULE_ALGORITHM_NAME = ALGORITHM[1]
ALGORITHM_NAME = ALGORITHM[2]

# Parameters
if DATASET[0] == 'Brats':

    MODULE_DATASET_NAME = DATASET[1]
    DATASET_NAME = DATASET[2]
    num_classes = 2 # 4                                            # 2: tumor, background   4: background, peritumoral edema, enhancing tumor, tumor core
    number_medical_exams = 1  # 4                                  # Flair, T1, T1ce, T2
    NUM_CLIENTS = 10                                                # 4_th is Athens, 5_th is centralized
    NUM_CLIENTS_non_iid = 0
    NUM_CLIENTS_iid = NUM_CLIENTS - NUM_CLIENTS_non_iid
    CLIENTS_SELECTED = NUM_CLIENTS # C                              # Clients at each Federated_round
    BATCH_SIZE = 14   # 16 14                                             # B
    SHUFFLE_BUFFER = 1024 # 1024
    CACHE_DATASET_TRAIN = 0
    CACHE_DATASET_VALID = 0
    PREFETCH_BUFFER = tf.data.AUTOTUNE
    NUM_ROUNDS = 800 # 150 1,  400 1/4, 800+ 1/8                                              # Number of Federated_round
    clients_id = range(NUM_CLIENTS)
    PATIENCE =  NUM_ROUNDS                                           # Early stopping in Federated Rounds
    MAX_MODEL_SIZE =  250 * 10**6
    # 74 * 10**3
    # 250 * 10**6
    # 55 * 10**3
    NUM_PARTS_DATASET = 1
    PERCENTAGE_DATASET = 1
    INCOMING_CLIENTS = False

    TYPES_DATASET_COMPOSITION = {
                        0: 'LGG',
                        1: 'HGG',
                        2: 'athens',
                        3: 'HGG_LGG',
                        4: 'HGG_LGG_athens'
                    }
    DATASET_COMPOSITION = TYPES_DATASET_COMPOSITION[4]

elif DATASET[0] == 'Mnist':

    num_classes = 10
    NUM_CLIENTS = 3                                                # K
    NUM_CLIENTS_non_iid = 0
    NUM_CLIENTS_iid = NUM_CLIENTS - NUM_CLIENTS_non_iid
    CLIENTS_SELECTED = NUM_CLIENTS # C                              # Clients at each Federated_round
    BATCH_SIZE = 20                                                 # B
    SHUFFLE_BUFFER = 100
    PREFETCH_BUFFER = 10
    NUM_ROUNDS = 4                                                  # Number of Federated_round
    clients_id = range(NUM_CLIENTS)
    PATIENCE =  12                                          # Early stopping in Federated Rounds
    MAX_MODEL_SIZE =  250 * 10**6
    # 74 * 10**3
    # 250 * 10**6
    # 55 * 10**3
    NUM_PARTS_DATASET = 1
    PERCENTAGE_DATASET = 1
    INCOMING_CLIENTS = False

elif DATASET[0] == 'Mnist_non_iid':
 
    num_classes = 10
    NUM_CLIENTS = 3                                             # K
    NUM_CLIENTS_non_iid = 0
    NUM_CLIENTS_iid = NUM_CLIENTS - NUM_CLIENTS_non_iid
    CLIENTS_SELECTED = NUM_CLIENTS # C                              # Clients at each Federated_round
    BATCH_SIZE = 20                                                 # B
    SHUFFLE_BUFFER = 100
    PREFETCH_BUFFER = 10
    NUM_ROUNDS = 10    # 600                                              # Number of Federated_round
    clients_id = range(NUM_CLIENTS)
    PATIENCE =  NUM_ROUNDS                                           # Early stopping in Federated Rounds
    MAX_MODEL_SIZE =  250 * 10**6
    # 74 * 10**3
    # 250 * 10**6
    # 55 * 10**3
    NUM_PARTS_DATASET = 1#3                                         #Ãƒâ€šÃ‚Â Number of pieces in which to divide the dataset.
                                                                    # Every half of total federated rounds, we increse the dataset of each client of 1/NUM_PARTS_DATASET % 
    PERCENTAGE_DATASET = 1#1/6                                       # Use a percentage of the whole dataset
    INCOMING_CLIENTS = 0


#########################THIRD

# Create random Proximity Matrix (Connected)
# PROXIMITY_MATRIX = np.random.randint(2,size=(NUM_CLIENTS,NUM_CLIENTS))
# PROXIMITY_MATRIX = np.round((PROXIMITY_MATRIX + PROXIMITY_MATRIX.T)/2) + np.identity(NUM_CLIENTS) 
# PROXIMITY_MATRIX [PROXIMITY_MATRIX == 2] = 1
# for i in range(NUM_CLIENTS-1):
#     PROXIMITY_MATRIX[i+1][i] = 1
#     PROXIMITY_MATRIX[i][i+1] = 1
# PROXIMITY_MATRIX [PROXIMITY_MATRIX == 2] = 1

# Create Fully-Connected Proximity Matrix
PROXIMITY_MATRIX = np.ones((NUM_CLIENTS,NUM_CLIENTS))
# print(PROXIMITY_MATRIX)

# Consensus parameters
EPSILON = 0.3    # 0.3 MNIST                                                  # Consensus step size
Q = 0.99         # 0.99 MNIST                                                # Hyperparameter in MEWMA for gradients
Beta = 1# 10**(-3) #200                                         # Mixing weights for the gradients
Gossip = 0                                                      # 1 if Gossip is enabled
COORD_SLEEP_TIME = 497*1/16                                            # [s]

# PS parameters
NUM_EPOCHS = 1    # 2 MNIST                                              # Local_epoch per Federated_round
EPSILON_GLOBAL_MODEL = 1                                        # Hyperparameter in MEWMA for global_model
SERVER_SLEEP_TIME = 497*1/4 # 2 MNIST                                           # In case of Asynchronous Server [s]

# Fed Adp parameters
ALPHA = 10   # 5 MNIST

# Fed Prox parameters
MU = 0     # 10 MNIST

# Optimization parameters
if DATASET[0] == 'Brats':

    #LR = tf.keras.optimizers(2e-4, decay_steps = )              # Learning rate (Initial)
    LR = 1e-4                       # centralized  2e-4
    if ALGORITHM[0] == 'FedProx' or ALGORITHM[0] == 'FedAdpProx':
        OPTIMIZER = Adam                     # Optimization algorithm for FedProx (PerturbedGradientDescent, AMSGrad, Adam)
    else:
        OPTIMIZER = tf.keras.optimizers.Adam                      # Optimization algorithm 
    LOSS = custom_loss                                          # Loss to be minimized
    # LOSS = generalized_dice # generalized_dice_loss
    # LOSS = combined_dice_ce_loss # dice_coef_loss
    # LOSS = gen_dice_loss

    if num_classes == 2:
        METRICS = [dice_whole_metric]  # Accuracy measured
    if num_classes == 4:
        METRICS = [dice_whole_metric,dice_core_metric,dice_en_metric]  # Accuracy measured

    # LOSS = tf.keras.losses.SparseCategoricalCrossentropy()      # Loss to be minimized
    # METRICS = [meanIoU]                                         # Accuracy measured

    VALIDATION_ROUNDS = range(0, NUM_ROUNDS, 1)

elif DATASET[0] == 'Mnist' or DATASET[0] == 'Mnist_non_iid':

    LR = 1e-3                                                   # Learning rate (Initial)
    if ALGORITHM[0] == 'FedProx' or ALGORITHM[0] == 'FedAdpProx':
        OPTIMIZER = PerturbedGradientDescent                     # Optimization algorithm for FedProx 
    else:
        OPTIMIZER = tf.keras.optimizers.Adam                      # Optimization algorithm 
    LOSS = tf.keras.losses.CategoricalCrossentropy()            # Loss to be minimized
    METRICS = [tf.keras.metrics.CategoricalAccuracy()]#,tf.keras.metrics.Precision(),tf.keras.metrics.Recall()]   # Accuracy measured

    VALIDATION_ROUNDS = range(0, NUM_ROUNDS, 1)
