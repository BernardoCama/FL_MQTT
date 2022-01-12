#!/usr/bin/env python
# coding: utf-8

# PS-S-C-S
# Server counterpart in FL with parameter server

# Import parameters
import os

from tensorflow.python import training
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from Classes.Params.param import *
try:
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass  
from Classes.Auxiliary.Cut_model import Cut_model
from Classes.Auxiliary.Launch_MQTT_broker import Launch_MQTT_broker
from Classes.Auxiliary.Stop_train import Check_stop_train
from Classes.Auxiliary.Recover_samples import Recover_samples
from Classes.Auxiliary.Ping import Ping

# Import dataset
DATASET = getattr(importlib.import_module(MODULE_DATASET_NAME), DATASET_NAME)

# Initialize dataset
dataset = DATASET()

num_images_train = dataset.return_num_images_train()
num_images_test = dataset.return_num_images_test()
num_images_valid = dataset.return_num_images_valid()
input_shape = dataset.return_input_shape()

# Import model
MODEL = getattr(importlib.import_module(MODULE_MODEL_NAME), MODEL_NAME)

# Initialize model
model = MODEL(input_shape)

# Import algorithm
ALGORITHM = getattr(importlib.import_module(MODULE_ALGORITHM_NAME), ALGORITHM_NAME)

# Initialize algorithm
algorithm = ALGORITHM()


if __name__ == '__main__':

    print('Start server')

    # Inizialize MQTT server
    Launch_MQTT_broker()

    # Initialize the model
    model_server = model.return_model()

    # Number of layers in the model
    num_layers = len(model_server.weights)
    
    ##############################################################################################################
    algorithm.initialize_local_variables_Server(model_server, num_layers, NUM_CLIENTS)

    # Save weights locally
    WEIGHTS_GLOBAL = algorithm.WEIGHTS_GLOBAL
    WEIGHTS_CLIENTS = algorithm.WEIGHTS_CLIENTS

    # Save gradients locally
    GRADIENTS_GLOBAL = algorithm.GRADIENTS_GLOBAL
    GRADIENTS_CLIENTS = algorithm.GRADIENTS_CLIENTS
    ##############################################################################################################

    # Compute model size
    size_model = sys.getsizeof(zlib.compress(cPickle.dumps(WEIGHTS_GLOBAL)))
    print('Model memory size in B: {}, KB: {}, MB: {}'.format(size_model, size_model/10**3, size_model/10**6))

    cutting_points, MODEL_SNIPPETS = Cut_model(size_model, MAX_MODEL_SIZE, num_layers, WEIGHTS_GLOBAL)


    # Active Clients and samples
    Working_clients = {}
    Ping('Server', NUM_CLIENTS, Working_clients, Last_seen_time_topic, broker_address, MQTT_port, AUTH_, TLS_)
    Recover_samples('Server', num_images_train, num_images_test, num_images_valid, NUM_CLIENTS, samples_topic, broker_address, MQTT_port, AUTH_, TLS_)
    time.sleep(2)

    # Upload of weights
    messages = []
    latest_list_of_requested_clients = list(Working_clients.keys())

    for snippet in range(MODEL_SNIPPETS):
        payload = zlib.compress(cPickle.dumps({'weights': [WEIGHTS_GLOBAL[layer] for layer in range(cutting_points[snippet], cutting_points[snippet+1])],
                                    'snippet': snippet,
                                    'NUM_EPOCHS': NUM_EPOCHS,
                                    'Learning_rate': LR,
                                    'requested_clients': latest_list_of_requested_clients,
                                    }))
        messages.append({'topic':server_weights_topic + '/{}'.format(snippet), 'payload': payload, 'qos': 2, 'retain': True})

    publish.multiple(messages, hostname=broker_address, port=MQTT_port, client_id="Server", keepalive=100,
                    will=None, auth=AUTH_, tls=TLS_, protocol=mqtt.MQTTv311, transport="tcp")

    print(f'Weights Uploaded, waiting for the following Clients: {latest_list_of_requested_clients}')

    # Metrics of Loss and Accuracy of each Client for each round  
    partial_metrics = [[0 for col in range(NUM_CLIENTS)] for row in range(NUM_ROUNDS)]


    # Clients for round 0
    sample_clients = {elem:Working_clients[elem] for elem in random.sample(list(Working_clients), min(CLIENTS_SELECTED, len(Working_clients)))}     

    roundStart = time.time()

    # Result of training
    result = []
    timings = {'upload': [[0 for col in range(NUM_CLIENTS+5)] for row in range(NUM_ROUNDS)],
                'validation':[[0 for col in range(NUM_CLIENTS+5)] for row in range(NUM_ROUNDS)],
                'training':[[0 for col in range(NUM_CLIENTS+5)] for row in range(NUM_ROUNDS)],
                'download':[[0 for col in range(NUM_CLIENTS+5)] for row in range(NUM_ROUNDS)]}

    while 1:

        # Receive weights of current round - Blocking state
        print(f'Server waiting client weights - waiting for {len(latest_list_of_requested_clients)} messages. ')

        messages = subscribe.simple(client_weights_topic + '/#', qos=2, msg_count=len(latest_list_of_requested_clients)*MODEL_SNIPPETS, retained=False, hostname=broker_address,
                port=MQTT_port, client_id="Server", keepalive=10, will=None, auth=AUTH_, tls=TLS_,
                protocol=mqtt.MQTTv311)

        print(f'Server received client weights ({len([messages])} msgs)')

        for message in (messages if type(messages) is list else [messages]):

            body = cPickle.loads(zlib.decompress(message.payload))

            client_ = body['client']
            weights = body['weights']
            round_ = body['round']
            metric_ = body['metrics']
            snippet_ = body['snippet']
            Stop_Training = body['Stop_Training']
            try:
                training_ = body['timing']['training']
            except:
                training_ = -1
            try:
                validation_ = body['timing']['validation']
            except:
                validation_ = -1
            try:
                upload_ = body['timing']['upload']
            except:
                upload_ = -1
            try:
                download_ = body['timing']['download']
            except:
                download_ = -1

            # print('Server received from client: {} snippet: {}'.format(client_, snippet_))

            i = 0
            for layer in range(cutting_points[snippet_], cutting_points[snippet_+1]):
                WEIGHTS_CLIENTS[client_][layer] = weights[i]
                i = i + 1                

            print('Assign metric of round: {} and client: {}'.format(round_, client_))

            # If the Client is selected in this round
            if client_ in sample_clients:
                partial_metrics[round_][client_] = metric_
                timings['upload'][round_][client_] = upload_
                timings['download'][round_][client_] = download_
                timings['training'][round_][client_] = training_
                timings['validation'][round_][client_] = validation_

            else:
                partial_metrics[round_][client_] = -1
                timings['upload'][round_][client_] = -1
                timings['download'][round_][client_] = -1
                timings['training'][round_][client_] = -1
                timings['validation'][round_][client_] = -1

            # If Client finished training
            if Stop_Training:

                Working_clients[client_] = 0

                for i in range(round_ + 1, NUM_ROUNDS):

                    partial_metrics[i][client_] = - 1
                    timings['upload'][i][client_] = -1
                    timings['download'][i][client_] = -1
                    timings['training'][i][client_] = -1
                    timings['validation'][i][client_] = -1

        # End round
        if not 0 in [partial_metrics[round_][working_client] for working_client in latest_list_of_requested_clients]:   # with index belonging to working clients

            # Show timing information for the round
            roundEnd = time.time()
            elapsed = ( roundEnd - roundStart) 
            print("round {} took {:.4} seconds".format(round_, elapsed))

            # for i in range(round_ + 1):
            #     for j in range(NUM_CLIENTS):
            #         print('round: {} and client: {} metric: {}'.format(i, j, partial_metrics[i][j]))

            try:
                mean_accuracy = np.mean(np.squeeze(np.array([[acc for acc in elem['accuracy']] for elem in list(filter(lambda num: num != -1 and num != 0, partial_metrics[round_]))])), 0)
            except:
                mean_accuracy = np.mean(np.squeeze(np.array([[acc for acc in elem['accuracy']] for elem in list(filter(lambda num: num != -1 and num != 0, partial_metrics[round_]))])))
            mean_loss = np.mean([elem['loss'][-1] for elem in list(filter(lambda num: num != -1 and num != 0, partial_metrics[round_])) ])  
            print('Mean Accuracy Round {}: {}, Mean Loss: {}\n\n'.format(round_, [elem for elem in (mean_accuracy if type(mean_accuracy) is list else [mean_accuracy])], mean_loss))

            # Save statistics
            result.append({'round': round_, 'time_round': elapsed, 'mean_accuracy': [mean_accuracy], 'mean_loss': copy.copy(mean_loss), 'timings':copy.copy(timings)})
            np.save(cwd + '/result.npy', result, allow_pickle = True)

##############################################################################################################
            algorithm.update_weights_Server(round_, NUM_CLIENTS, num_images_train, sample_clients, CLIENTS_SELECTED, num_layers, WEIGHTS_GLOBAL, WEIGHTS_CLIENTS, GRADIENTS_GLOBAL, GRADIENTS_CLIENTS)

            WEIGHTS_GLOBAL = algorithm.WEIGHTS_GLOBAL
            WEIGHTS_CLIENTS = algorithm.WEIGHTS_CLIENTS

            GRADIENTS_GLOBAL = algorithm.GRADIENTS_GLOBAL
            GRADIENTS_CLIENTS = algorithm.GRADIENTS_CLIENTS
##############################################################################################################

            if round_ == NUM_ROUNDS - 1 or Check_stop_train(params_topic, broker_address, MQTT_port, AUTH_, TLS_):
                
                print('End training\n')
                break

            # If could come new Clients
            if param.INCOMING_CLIENTS:
                # Check if there is a new Client
                import Classes.Params.param_mutable as param_mutable
                param_mutable.check_new_client()
                # If there is a new Client
                if NUM_CLIENTS != param_mutable.NUM_CLIENTS:
##############################################################################################################
                    from Classes.Auxiliary.NewComer import NewClient
                    new_var = NewClient()
                    new_var.update_variables(round_ = round_ if round_ is not None else None, 
                                            client_ = client_ if client_ is not None else None,
                                            NUM_ROUNDS = NUM_ROUNDS, 
                                            NUM_CLIENTS = param_mutable.NUM_CLIENTS,
                                            CLIENTS_SELECTED = CLIENTS_SELECTED if CLIENTS_SELECTED is not None else None,
                                            Working_clients = Working_clients if Working_clients is not None else None,
                                            PROXIMITY_MATRIX = PROXIMITY_MATRIX if PROXIMITY_MATRIX is not None else None, 
                                            WEIGHTS_CLIENTS = WEIGHTS_CLIENTS if WEIGHTS_CLIENTS is not None else None, 
                                            GRADIENTS_CLIENTS = GRADIENTS_CLIENTS if GRADIENTS_CLIENTS is not None else None, 
                                            partial_metrics = partial_metrics if partial_metrics is not None else None, 
                                            theta_tilde = algorithm.theta_tilde if hasattr(algorithm, 'theta_tilde') else None, 
                                            DATASET = DATASET if DATASET is not None else None)

                    NUM_CLIENTS = new_var.NUM_CLIENTS
                    CLIENTS_SELECTED = new_var.CLIENTS_SELECTED
                    Working_clients = new_var.Working_clients
                    WEIGHTS_CLIENTS = new_var.WEIGHTS_CLIENTS
                    GRADIENTS_CLIENTS = new_var.GRADIENTS_CLIENTS
                    theta_tilde = new_var.theta_tilde
                    partial_metrics = new_var.partial_metrics
                    PROXIMITY_MATRIX = new_var.PROXIMITY_MATRIX
                    NEIGHBORS = new_var.NEIGHBORS
                    dataset = new_var.dataset
                    num_images_train = new_var.num_images_train
##############################################################################################################
                    # Send a message with retain_flag to True, to give info to the new Client
                    messages = []
                    for snippet in range(MODEL_SNIPPETS):
                        payload = zlib.compress(cPickle.dumps({'NUM_EPOCHS': NUM_EPOCHS,
                                                'Learning_rate': LR,
                                                'round': round_}))
                        messages.append({'topic':client_weights_topic + '/{}/{}'.format(NUM_CLIENTS - 1, snippet), 'payload': payload, 'qos': 2, 'retain': True})
                    publish.multiple(messages, hostname=broker_address, port=MQTT_port, client_id="Server", keepalive=10,
                                    will=None, auth=AUTH_, tls=TLS_, protocol=mqtt.MQTTv311, transport="tcp")

                    # Upload Server weights with retain_flag to True, to give weights to the new Client
                    retain_flag = True
                    
                else:
                    retain_flag = False
            else:
                retain_flag = False

            latest_list_of_requested_clients = list(Working_clients.keys())
            print(f'Server publishing new global weights for {len(latest_list_of_requested_clients)} clients...')
            messages = []
            for snippet in range(MODEL_SNIPPETS):
                payload = zlib.compress(cPickle.dumps({'weights': [WEIGHTS_GLOBAL[layer] for layer in range(cutting_points[snippet], cutting_points[snippet+1])],
                                        'snippet': snippet,
                                        'NUM_EPOCHS': NUM_EPOCHS,
                                        'Learning_rate': LR,
                                        'round': round_,
                                        'requested_clients': latest_list_of_requested_clients,
                                        }))
                messages.append({'topic':server_weights_topic + '/{}'.format(snippet), 'payload': payload, 'qos': 2, 'retain': True})

            publish.multiple(messages, hostname=broker_address, port=MQTT_port, client_id="Server", keepalive=10,
                            will=None, auth=AUTH_, tls=TLS_, protocol=mqtt.MQTTv311, transport="tcp")

            print(f'Server published new global weights for these clients: {latest_list_of_requested_clients}')

            # Clients for new round
            sample_clients = {elem:Working_clients[elem] for elem in random.sample(list(Working_clients), min(CLIENTS_SELECTED, len(Working_clients)))}        

            roundStart = time.time()

        # If all clients finished training
        if len(Working_clients)  == 0 or Check_stop_train(params_topic, broker_address, MQTT_port, AUTH_, TLS_):

            print('End training\n')
            break 
