import os
import sys
import inspect
import numpy as np
from glob import glob
from shutil import copyfile
import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
import paho.mqtt.subscribe as subscribe
import _pickle as cPickle
import zlib
import json

# 1 to start the training
# 0 NOT to start the training
TRAIN_FLAG = 0

# 1 to stop the training
TRAIN_STOP = 1

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
cwd = currentdir

# MQTT Topics and Broker address  
with open(os.path.join(cwd, 'MQTT_broker_config.json')) as f:
    config = json.load(f)

REMOTE = 1                                                 # Test local or remote
TLS_USAGE = 1

if not REMOTE:
    broker_address = '127.0.0.1'
else:
    broker_address = config['broker_address']

if not TLS_USAGE:
    MQTT_port = 1885   
    TLS_ = None       
    AUTH_ = None   
else:
    MQTT_port = config['MQTT_port']  # Address of the MQTT broker
    TLS_ = config['TLS_']       
    TLS_['tls_version'] = mqtt.ssl.PROTOCOL_TLS
    AUTH_ = config['AUTH_']  


server_weights_topic = config["Name_of_federation"] + 'server_weights'
client_weights_topic = config["Name_of_federation"] + 'client_weights'
params_topic =  config["Name_of_federation"] + 'params'
samples_topic = config["Name_of_federation"] + 'samples'
Last_seen_time_topic = config["Name_of_federation"] + 'Last_seen_time' 

# Param file
param_file = os.path.join(cwd, 'Classes/Params/param.py')

# Read param file
with open(param_file, 'r') as file:
    filedata = file.read()

# Publish param file
messages = []
payload = zlib.compress(cPickle.dumps({'TRAIN_FLAG': TRAIN_FLAG,
                                        'TRAIN_STOP': TRAIN_STOP,
                                        'params': filedata
}))                                   
messages.append({'topic':params_topic, 'payload': payload, 'qos': 2, 'retain': True})
publish.multiple(messages, hostname=broker_address, port=MQTT_port, client_id="Share_params", keepalive=10,
                will=None, auth=AUTH_, tls=TLS_, protocol=mqtt.MQTTv311, transport="tcp")

print('Parameters Uploaded\n')
