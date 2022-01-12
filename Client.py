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
import time
import platform
import re
import json
from Classes.Auxiliary.Ping import Ping

def waitingAnimation(n):
    n = n%3+1
    dots = n*'.'+(3-n)*' '
    sys.stdout.write('\r Waiting '+ dots)
    sys.stdout.flush()
    time.sleep(0.5)
    return n

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
cwd = currentdir

# Operating System
OPERATING_SYSTEM = platform.system()
if OPERATING_SYSTEM == 'Darwin':
    OPERATING_SYSTEM = 'MACOS'
    import appscript
elif OPERATING_SYSTEM == 'Windows':
    OPERATING_SYSTEM = 'WINDOWS'
elif OPERATING_SYSTEM == 'Linux':
    OPERATING_SYSTEM = 'UBUNTU'

# Conda environment name
try:
    CONDA_ENV = os.environ['CONDA_DEFAULT_ENV'] 
except:
    CONDA_ENV = sys.executable.split(os.sep)[3]

# Python exexcutable
PYTHON_CONDA_DIR = sys.executable

# MQTT Topics and Broker address
with open(os.path.join(cwd, 'MQTT_broker_config.json')) as f:
    config = json.load(f)
broker_address = config['broker_address']      # Address of the MQTT broker
MQTT_port = config['MQTT_port']   
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

# ID of the client
filename = inspect.stack()[0].filename.split(os.sep)[-1]
# client_ = re.findall('\d+',filename)
# client_ = int(client_[0])

if len(sys.argv) < 2:
    print(f"ERROR! Missing argument - client-ID. \n Usage: \n\t {sys.argv[0]} <client-id>")
    quit(3)

client_ = int(sys.argv[1])

n = 0
while True:

    try: 

        # Notify the Client
        Ping(client_, -1, {}, Last_seen_time_topic, broker_address, MQTT_port, AUTH_, TLS_, Stop_Training = 1)

        # Receive parameters    
        messages = subscribe.simple(params_topic, qos=2, msg_count=1, retained=True, hostname=broker_address,
                            port=MQTT_port, client_id="Client" + str(client_), keepalive=10, will=None, auth=AUTH_, tls=TLS_,
                            protocol=mqtt.MQTTv311)
        body = cPickle.loads(zlib.decompress(messages.payload))

        if body['TRAIN_FLAG']:
            filedata = body['params']

            # Write param file
            with open(param_file, 'w') as file:
                file.write(filedata)

            # Not secure
            filedata = filedata.split('\n')
            for line in filedata:
                if 'import' in line:
                    filedata.remove(line)
            filedata = '\n'.join(filedata)


            filedata = filedata.split('#########################FIRST')      
            filedata = filedata[1]
            filedata = filedata.split('#########################SECOND')
            filedata = filedata[1]
            
            try:
                exec(filedata)
            except:
                pass

            # Launch Client silently
            # client_filename = os.path.join(cwd, 'Client_{}_iid_{}_MQTT.py'.format(client_, ARCHITECTURE[0]))
            client_filename = os.path.join(cwd, f'Client_iid_{ARCHITECTURE[0]}_MQTT.py')

            # if not os.path.exists(client_filename):
            #     copyfile(os.path.join(cwd, 'Client_{}_iid_{}_MQTT.py'.format(0, ARCHITECTURE[0])), client_filename)
            try:
                Ping(client_, -1, {}, Last_seen_time_topic, broker_address, MQTT_port, AUTH_, TLS_)
                res = os.system( PYTHON_CONDA_DIR + ' ' + client_filename + ' ' + str(client_))
            except:
                print('os.system Error\n')
                Ping(client_, -1, {}, Last_seen_time_topic, broker_address, MQTT_port, AUTH_, TLS_, Stop_Training = 1)

    except KeyboardInterrupt:
        
        try:
            Ping(client_, -1, {}, Last_seen_time_topic, broker_address, MQTT_port, AUTH_, TLS_, Stop_Training = 1)
        except:
            pass

    time.sleep(10)
    n = waitingAnimation(n)