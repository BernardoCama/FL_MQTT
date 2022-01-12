# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from Classes.Params.param import *

from glob import glob
from shutil import copyfile
if OS == 'MACOS':
        import mac_tag


print(cwd)
Coordinator_file = cwd + '/Coordinator_Consensus_MQTT.py'
clients_list_iid = glob(cwd  + '/Client*iid_Consensus_MQTT.py')
clients_list_non_iid = glob(cwd  + '/Client*non_iid_Consensus_MQTT.py')
if OS == 'MACOS':
        appscript.app('Terminal').do_script("conda activate " + conda_env + "; python " + Coordinator_file + ";") 
elif OS == 'UBUNTU':
        os.system("gnome-terminal -e 'bash -c \"source ~/anaconda3/etc/profile.d/conda.sh;conda activate " + conda_env + "; python \"" + Coordinator_file + "\" exec bash\"'")
elif OS == 'WINDOWS':
        os.system('start cmd /k ' + python_conda_dir + ' ' + Coordinator_file)
time.sleep(10)

for client_ in range(0, NUM_CLIENTS_iid):

        if OS == 'MACOS' or OS == 'UBUNTU':
                if cwd + '/Client_{}_iid_Consensus_MQTT.py'.format(client_) not in clients_list_iid:
                        copyfile(cwd + '/Client_iid_Consensus_MQTT.py', cwd + '/Client_{}_iid_Consensus_MQTT.py'.format(client_))
                        if OS == 'MACOS':
                                mac_tag.add(["CFA"],[cwd + '/Client_{}_iid_Consensus_MQTT.py'.format(client_)])
                        # print('not client:{}'.format(client_))
        elif OS == 'WINDOWS':
                if cwd + '\\Client_{}_iid_Consensus_MQTT.py'.format(client_) not in clients_list_iid:
                        copyfile(cwd + '\\Client_iid_Consensus_MQTT.py', cwd + '\\Client_{}_iid_Consensus_MQTT.py'.format(client_))

        if OS == 'MACOS':
                appscript.app('Terminal').do_script("conda activate " + conda_env + "; python " + cwd + '/Client_{}_iid_Consensus_MQTT.py'.format(client_) + ";") 
        elif OS == 'UBUNTU':
                os.system("gnome-terminal -e 'bash -c \"source ~/anaconda3/etc/profile.d/conda.sh;conda activate " + conda_env + "; python \"" + cwd + '/Client_{}_iid_Consensus_MQTT.py'.format(client_) + "\" exec bash\"'")
        elif OS == 'WINDOWS':    
                os.system('start cmd /k ' + python_conda_dir + ' ' + cwd + '\\Client_{}_iid_Consensus_MQTT.py'.format(client_))


for client_ in range(NUM_CLIENTS_iid, NUM_CLIENTS_iid + NUM_CLIENTS_non_iid):
    
        if OS == 'MACOS' or OS == 'UBUNTU':
                if cwd + '/Client_{}_non_iid_Consensus_MQTT.py'.format(client_) not in clients_list_iid:
                        copyfile(cwd + '/Client_non_iid_Consensus_MQTT.py', cwd + '/Client_{}_non_iid_Consensus_MQTT.py'.format(client_))
                        if OS == 'MACOS':
                                mac_tag.add(["CFA"],[cwd + '/Client_{}_non_iid_Consensus_MQTT.py'.format(client_)])
                        # print('not client:{}'.format(client_))
        elif OS == 'WINDOWS':
                if cwd + '\\Client_{}_non_iid_Consensus_MQTT.py'.format(client_) not in clients_list_iid:
                        copyfile(cwd + '\\Client_non_iid_Consensus_MQTT.py', cwd + '\\Client_{}_non_iid_Consensus_MQTT.py'.format(client_))

        if OS == 'MACOS':
                appscript.app('Terminal').do_script("conda activate " + conda_env + "; python " + cwd + '/Client_{}_non_iid_Consensus_MQTT.py'.format(client_) + ";") 
        elif OS == 'UBUNTU':
                os.system("gnome-terminal -e 'bash -c \"source ~/anaconda3/etc/profile.d/conda.sh;conda activate " + conda_env + "; python \"" + cwd + '/Client_{}_non_iid_Consensus_MQTT.py'.format(client_) + "\" exec bash\"'")
        elif OS == 'WINDOWS':    
                os.system('start cmd /k ' + python_conda_dir + ' ' + cwd + '\\Client_{}_non_iid_Consensus_MQTT.py'.format(client_))
