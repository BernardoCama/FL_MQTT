# FL MQTT
Code of the paper: "Decentralized Federated Learning for Healthcare Networks: A Case Study on Tumor Segmentation".




---
## Setup


### Environment
Install the conda environment selecting the correct OS from the [folder](https://github.com/BernardoCama/FL_MQTT/tree/main/Environments):

```
conda env create -f environment_windows.yml
```

If it gives problem during the procedure, install manually all the following packages in the [**miniconda**](https://docs.conda.io/en/latest/miniconda.html) environment:

```
conda install nomkl
conda install -c intel scikit-learn
conda install seaborn
conda install psutil
conda install -c simpleitk simpleitk 
pip3 install dltk
pip3 install dill
pip3 install pydicom
pip3 install opencv-python
pip3 install gdcm
pip3 install pylibjpeg
pip3 install pylibjpeg-libjpeg
# to solve binary incompatibility
pip3 install --upgrade tensorflow
pip3 uninstall numpy
pip3 install numpy
pip3 install nvidia-ml-py3
pip3 install paho-mqtt
```


## Launch MQTT broker (without TLS encryption)

Stopping and removing broker from [**Docker**](https://docs.docker.com/get-docker/):
```
docker stop hivemq-ce
docker rm hivemq-ce
```

Launch MQTT broker inside docker creating two mapping rules (outside:inside Docker):
- 1883:1883 for outside Clients.
- 11883:1883 for Clients in same machine of the MQTT broker.
```
docker run --name hivemq-ce -e HIVEMQ_LOG_LEVEL=INFO -d -p 1883:1883 -p 11883:1883 hivemq/hivemq-ce
```
Verify correct launch of the MQTT broker:
```
docker container ls -a
```

If you want the MQTT broker to be publicly reachable, please use TLS encryption and an Authentication mechanism.
Perform a port forwarding towards open internet and use a static IP for the broker network or a [**DynamicIP tool**](https://www.noip.com).

## Specify MQTT broker configuration
The MQTT broker configuration must be completed in 
each PC that will be connected to the MQTT broker:
thus each computer containing the PS, the clients and the 
[**Command and Control tool**](https://github.com/BernardoCama/FL_MQTT/blob/main/Share_params.py).
You can specify:
   - broker_address: address of the MQTT broker that can be publicly reachable or through SSH.
   - MQTT_port: port of the MQTT broker.
   - TLS_: path to the certificate .crt if TLS set to true.
   - AUTH_: username and password for the authentication to the MQTT broker.
   - Name_of_federation: name of the federation we want to connect to.


## MQTT broker not publicly reachable - SSH Connection
If the MQTT broker is inside the CERN network, 
we take advantage of the tunneling SSH CERN service called $lxtunnel$, which performs a port-forwarding of the incoming connections towards the listening broker inside the Docker Engine at a specific port.

### Windows
On Putty create tunnel to lxtunnel.cern.ch port 22 with tunnel:
   - src: 8022 to 128.141.183.190:22 
   - src: 11883 to 128.141.183.190:1883 (hivemq-ce broker listening)
   
The first rule is used to connect via SSH to the Pegasus machine (128.141.183.190) and the second rule to forward the MQTT messages.


### Ubuntu
From the terminal:
```
ssh -L 8022:128.141.183.190:22 [cern_username]@lxtunnel.cern.ch
ssh -L 11883:128.141.183.190:1883 [cern_username]@lxtunnel.cern.ch
```


### Test MQTT reachability
Install MQTT Client as [**Mosquitto**](https://mosquitto.org/download/).

First Subscribe (first terminal)
```
mosquitto_sub -h localhost -p 11883 -t first_topic
```

Second Publish (second terminal)
```
mosquitto_pub -h localhost -p 11883 -m "hello" -t first_topic
```


## MQTT broker publicly reachable
Install MQTT Client as [**Mosquitto**](https://mosquitto.org/download/).

First Subscribe (first terminal)
```
mosquitto_sub -h broker_address -p 1883 -t first_topic
```

Second Publish (second terminal)
```
mosquitto_pub -h broker_address -p 1883 -m "hello" -t first_topic
```


---
## Folder organization



### Dataset organization
The number of images in each Client and the 
presence of a Client in the training, will be shared
through MQTT messages.

Each Client will have its dataset in the H[ClientID] folder:

REMOTE case:
- Client_i: 
    - Hi:
        - training:
            - Patient1
            - Patient2
            - ...
            - tfrecod file (created at the beginning of the training)
        - test:
            - Patient1
            - Patient2
            - ...
            - tfrecod file (created at the beginning of the training)
        - validation:
            - Patient1
            - Patient2
            - ...
            - tfrecod file (created at the beginning of the training)  



### Files organization
- [Classes](https://github.com/BernardoCama/FL_MQTT/tree/main/Classes): containing the optimizer, models, losses, metrics, dataset management, FL algorithms, parameter file and others.
- [Environments](https://github.com/BernardoCama/FL_MQTT/tree/main/Environments): containing the conda environment.
- [Share_params.py](https://github.com/BernardoCama/FL_MQTT/blob/main/Share_params.py): to load the parameters into the MQTT broker.
- [Server.py](https://github.com/BernardoCama/FL_MQTT/blob/main/Server.py), [Client.py](https://github.com/BernardoCama/FL_MQTT/blob/main/Client.py): to launch respectively the Server and Client.



---
## Dataset


### Download
The dataset of each specific Client can be dowloaded with these links:
- [H0](https://polimi365-my.sharepoint.com/:f:/g/personal/10584438_polimi_it/ElSESGSOSKZIuDKpy-gxj-MB5_8G9iGKz0_daI7pYPPpXw?e=oYiUDv
) (Politecnico/CNR PC)
- [H1](https://polimi365-my.sharepoint.com/:f:/g/personal/10584438_polimi_it/EjZ8dBinYjlPmeJD1wKQXRMB1nK5_Etk9_t1921w5CxMYQ?e=eIXbzd
) (Ioannis' PC)
- [H2](https://polimi365-my.sharepoint.com/:f:/g/personal/10584438_polimi_it/EsMDR-BkkTxHoBpV9lssQwMB6PTgMwHdFUDxvTn5o54Xqg?e=B3zp2Z
) (Roman's PC)
- [H3](https://polimi365-my.sharepoint.com/:f:/g/personal/10584438_polimi_it/EnrE_OYgl01GqNQ2ABKJDLEBhiCF1t0cDp5K0H9PaozicQ?e=U1wGjR
) (Pegasus PC)

The datasets are protected with a password.

To obtain the BraTS 2018 and BraTS 2020 you can refer to the pages:
[BraTS 2018](https://www.med.upenn.edu/sbia/brats2018/data.html)
[BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/data.html)




---
## Launch MQTT Clients



### Parameter Server
In the computer where you want to place the Parameter Server,
go into the folder FL_MQTT.

Activate the conda environment (in this case "tf_env"):
```
conda activate tf_env
```
In Windows, search for _Anaconda Powershell Prompt_ and then activate the environment.

Launch the Server:
```
python Server.py
```

This script will work as Server in the Server-based FL architecture and just as Collector of statistics in case of fully-decentralized architecture with Consensus driven algorithm.

The results of the training will be saved in the file _result.npy_ in the FL_MQTT folder.
The saved weights of the models at each epoch will be saved in the H[ClientID] folder inside _saved_weights_.

In the same computer where is placed the Parameter Server, you can have a visual representation of the validation accuracy by running the python file [plot.py](https://github.com/BernardoCama/FL_MQTT/blob/main/plot.py).
You can specify the global variables:
- DELETE: 1 if you want to remove the old _result.npy_ file.
- REAL_TIME = 1, TIME_ROUNDS = 0: to see the validation accuracy during the training in real time, with the federated rounds on the x axis.
- REAL_TIME = 0, TIME_ROUNDS = 0: to see the validation accuracy after the training, with the federated rounds on the x axis.
- REAL_TIME = 0, TIME_ROUNDS = 1: to see the validation accuracy after the training, with the seconds on the x axis.
```
conda run -n tf_env python plot.py
```



### Client of the Federation
Run the [Client](https://github.com/BernardoCama/FL_MQTT/blob/main/Client.py) with:
```
conda activate tf_env
python Client_ID.py [ClientID]
```

After launching the Server and Client scripts, they will show a _waiting message_ until the Command and Control message. 


---
## Parameters

### Select the settings
In the [param.py](https://github.com/BernardoCama/FL_MQTT/blob/main/Classes/Params/param.py) and in the [Share_params.py](https://github.com/BernardoCama/FL_MQTT/blob/main/Share_params.py) files, we can select the type of setting:
- REMOTE: 1 if the training is done in remote machines, 0 if locally.
- TLS_USAGE: 1 if we want to use TLS protocol.

Additionally in the [param.py](https://github.com/BernardoCama/FL_MQTT/blob/main/Classes/Params/param.py) file we can also launch locally the Mosquitto MQTT broker setting LAUNCH_MQTT_BROKER to 1 and specifying the broker dir in MQTT_broker_dir and the broker config file in MQTT_broker_config_file.

### Select the parameters
In the [param.py](https://github.com/BernardoCama/FL_MQTT/blob/main/Classes/Params/param.py) file, the main parameters to select are:
- DATASET = TYPES_DATASET[_to select_]: number 3 for medical data.                                      
- MODEL = TYPES_MODELS[_to select_]: number 5 for the U-net 2D.                                         
- ALGORITHM = TYPES_ALGORITHMS[_to select_]: the algorithms for the FL.                              
- ARCHITECTURE = TYPES_ARCHITECTURE[_to select_]: you can choose between:
    - 0: PS-S/C-S
    - 1: PS-A/C-S
    - 2: PS-A/C-A
    - 3: C/C-A

The possible types of models, dataset, algorithms and architecture can be seen above these constants.

After there are the parameters for the specific dataset:
- num_classes: number of output classes from the model.
- number_medical_exams: number of input channels in the model:
- NUM_CLIENTS: number of clients in the federation.
- CLIENTS_SELECTED: number of clients to be selected at each round by the Server.
- BATCH_SIZE 
- SHUFFLE_BUFFER: length of the shuffle buffer.
- CACHE_DATASET_TRAIN: 1 if we want to cache the entire training dataset in memory.
- CACHE_DATASET_VALID: 1 if we want to cache the entire validation dataset in memory.
- PREFETCH_BUFFER: number of batches to be prepared while the current one is being processed.
- NUM_ROUNDS: number of federated rounds.                                       
- PATIENCE: number of federated rounds without improving in the local loss function in order to apply early stopping.
- MAX_MODEL_SIZE: max MQTT message size.

Below there are the parameters for the Asynchronous architectures, as:

- COORD_SLEEP_TIME: time interval to check local round statistics for the collector of statistics in the fully-decentralized architecture.                           
- NUM_EPOCHS: number of local epochs to be performed in each Client.    
- SERVER_SLEEP_TIME: time interval to check for local model update in the PS-A-C-A and PS-S-C-A architectures.

Finally there are the parameters for the Non-IID algorithms and the specific losses, metrics and optimizer for the model:
- ALPHA: alpha hyper-parameter for FedAdp.
- MU: mu hyper-parameter for FedProx.
- LR: learning rate.
- OPTIMIZER: optimizer for the training.                   
- LOSS: loss for the training.
- METRICS: accuracy metrics.
- VALIDATION_ROUNDS: federated rounds at which do the local validation of the model. 



### Upload parameters
After selecting the parameters in the [param.py](https://github.com/BernardoCama/FL_MQTT/blob/main/Classes/Params/param.py) file, you can load them, from whatever PC (with SSH tunneling) to the MQTT broker by running 
the [Share_params.py](https://github.com/BernardoCama/FL_MQTT/blob/main/Share_params.py) with the global variable TRAIN_FLAG set to 0:

```
conda activate tf_env
python Share_params.py
```




---
## Training

### Start training
You can start the training by simply running again the [Share_params.py](https://github.com/BernardoCama/FL_MQTT/blob/main/Share_params.py) file with TRAIN_FLAG set to 1:

```
conda activate tf_env
python Share_params.py
```
After the training has started, you can reload the parameters with TRAIN_FLAG set to 0 so that, after the training has finished (reached the number of local epochs or due to early stopping), the MQTT Clients will go to sleep not using resources.


### Stop training
If for whatever reason, you need to stop the training, you can upload the parameters with TRAIN_STOP set to 1 and TRAIN_FLAG set to 0.




