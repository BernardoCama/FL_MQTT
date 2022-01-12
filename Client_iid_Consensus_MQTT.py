#!/usr/bin/env python
# coding: utf-8

# Client counterpart in FL with Consensus (NO TFF)

# Import parameters
from paho.mqtt import client
from Classes.Params.param import *
from Classes.Auxiliary.Cut_model import Cut_model
from Classes.Auxiliary.Save_model import Save_model
from Classes.Auxiliary.Stop_train import Check_stop_train
from Classes.Auxiliary.Recover_samples import Recover_samples
from Classes.Auxiliary.Ping import Ping
param.time.sleep(3)

# ID of the client
filename = inspect.stack()[0].filename.split(os.sep)[-1]
# client_ = re.findall('\d+',filename)
# client_ = int(client_[0])

if len(sys.argv) < 2:
    print(f"ERROR! Missing argument - client-ID. \n Usage: \n\t {sys.argv[0]} <client-id>")
    quit(3)

client_ = int(sys.argv[1])

time.sleep((random.randint(10,15))*3)

# Import dataset
DATASET = getattr(importlib.import_module(MODULE_DATASET_NAME), DATASET_NAME)

# Initialize dataset
dataset = DATASET()

num_images_train = dataset.return_num_images_train()
num_images_test = dataset.return_num_images_test()
num_images_valid = dataset.return_num_images_valid()
input_shape = dataset.return_input_shape()
Recover_samples(client_, num_images_train, num_images_test, num_images_valid, NUM_CLIENTS, samples_topic, broker_address, MQTT_port, AUTH_, TLS_)
print('input shape:{}'.format(input_shape))

# Import model
MODEL = getattr(importlib.import_module(MODULE_MODEL_NAME), MODEL_NAME)

# Initialize model
model = MODEL(input_shape)

# Import algorithm
ALGORITHM = getattr(importlib.import_module(MODULE_ALGORITHM_NAME), ALGORITHM_NAME)

# Initialize algorithm
algorithm = ALGORITHM()


if __name__ == '__main__':

    print('Start client: {}'.format(client_))

    # Initialize the model
    model_client = model.return_model()

    # Number of layers in the model
    num_layers = len(model_client.weights)

    # Save weights locally
    WEIGHTS_CLIENTS = [copy.copy(model_client.weights[layer].numpy()) for layer in range(num_layers)]

    # Compute model size
    size_model = sys.getsizeof(zlib.compress(cPickle.dumps(WEIGHTS_CLIENTS)))
    print('Model memory size in B: {}, KB: {}, MB: {}'.format(size_model, size_model/10**3, size_model/10**6))
    
    cutting_points, MODEL_SNIPPETS = Cut_model(size_model, MAX_MODEL_SIZE, num_layers, WEIGHTS_CLIENTS)

    # Take datasets
    if NUM_PARTS_DATASET != 1:
        train_dataset_preprocess = dataset.create_tf_dataset_for_client_train(client_, 0)
    else:
        train_dataset_preprocess = dataset.create_tf_dataset_for_client_train(client_)
    valid_dataset_preprocess = dataset.create_tf_dataset_for_client_valid(client_)
    test_dataset_preprocess = dataset.create_tf_dataset_for_client_test(client_)

    # Optimization params
    # Loss
    loss = LOSS

    # Learning rate
    lr = LR
    optimizer = OPTIMIZER(learning_rate=lr)

    # Validation metrics
    metrics = METRICS

    # Compile Model
    model_client.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Warm up
    model_client.fit(x=train_dataset_preprocess,          
                y=None, 
                epochs=NUM_EPOCHS, 
                steps_per_epoch=int(np.ceil(num_images_train[client_] / BATCH_SIZE)),
                verbose = 1)

    ##############################################################################################################
    algorithm.initialize_local_variables_Server(model_client, num_layers, NUM_CLIENTS)

    # Save weights locally for each Client
    WEIGHTS_CLIENTS = algorithm.WEIGHTS_CLIENTS

    # Save gradients locally for each Client
    GRADIENTS_CLIENTS = algorithm.GRADIENTS_CLIENTS
    ##############################################################################################################

    # Upload weights     
    messages = []
    for snippet in range(MODEL_SNIPPETS):
        payload = zlib.compress(cPickle.dumps({'weights': [copy.copy(WEIGHTS_CLIENTS[client_][layer]) for layer in range(cutting_points[snippet], cutting_points[snippet+1])],
                                'snippet': snippet, 
                                'client': client_,
                                'Stop_Training': False}))
        messages.append({'topic':client_weights_topic + '/{}/{}'.format(client_, snippet), 'payload': payload, 'qos': 2, 'retain': True})

    publish.multiple(messages, hostname=broker_address, port=MQTT_port, client_id="Client{}".format(client_), keepalive=60,
                    will=None, auth=AUTH_, tls=TLS_, protocol=mqtt.MQTTv311, transport="tcp")

    # Compute neighbors
    NEIGHBORS = np.where(PROXIMITY_MATRIX[client_, :] == 1)[0]
    NEIGHBORS = NEIGHBORS[NEIGHBORS!= client_]
    # print('Neighbors of client {} are: {}\n'.format(client_, NEIGHBORS))

    # If Gossip, select new neighbor
    # neighbors = NEIGHBORS if not Gossip else np.random.choice(NEIGHBORS, size=1, replace=False)
    neighbors = {}
    Ping('Client', NUM_CLIENTS, neighbors, Last_seen_time_topic, broker_address, MQTT_port, AUTH_, TLS_)
    Recover_samples('Client', num_images_train, num_images_test, num_images_valid, NUM_CLIENTS, samples_topic, broker_address, MQTT_port, AUTH_, TLS_)
    time.sleep(2)

    # Latest Validation losses for Early Stopping
    latest_losses = np.array([100.]*PATIENCE)
    Stop_Training = False

    # Result of training
    result = []
    
    # Recover round from neighbors
    messages = subscribe.simple(client_weights_topic + '/#', qos=2, msg_count=MODEL_SNIPPETS, retained=True, hostname=broker_address,
                        port=MQTT_port, client_id="Client" + str(client_), keepalive=10, will=None, auth=AUTH_, tls=TLS_,
                        protocol=mqtt.MQTTv311)
    for message in (messages if type(messages) is list else [messages]):
        body = cPickle.loads(zlib.decompress(message.payload))
        try:
            START_ROUND = body['round'] + 1
        except:
            START_ROUND = 0

    for round_ in range(START_ROUND, NUM_ROUNDS):

        roundStart = time.time() # len neighbors

        Ping(client_, NUM_CLIENTS, neighbors, Last_seen_time_topic, broker_address, MQTT_port, AUTH_, TLS_)

        # For each neighbor receive weights
        downloadStart = time.time()
        neighbors.pop(client_)
        for neighbor in list(neighbors):

            messages = subscribe.simple(client_weights_topic + '/{}/#'.format(neighbor), qos=2, msg_count=MODEL_SNIPPETS, retained=True, hostname=broker_address,
                            port=MQTT_port, client_id="Client" + str(client_), keepalive=60, will=None, auth=AUTH_, tls=TLS_,
                            protocol=mqtt.MQTTv311)
        
            for message in (messages if type(messages) is list else [messages]):
        
                body = cPickle.loads(zlib.decompress(message.payload))
                weights = body['weights']
                snippet_ = body['snippet']  
                Stop_Training_ = body['Stop_Training']
                print('round: {} client: {} received weights snippet: {} of neighbor: {}'.format(round_, client_, snippet_, neighbor))                         

                i = 0
                for layer in range(cutting_points[snippet_], cutting_points[snippet_+1]):
                    WEIGHTS_CLIENTS[neighbor][layer] = weights[i]
                    i = i + 1   

            # Remove neighbor if it has finished training
            # if Stop_Training_: 
            #    neighbors = neighbors[neighbors!=neighbor]

        downloadEnd = time.time()
        elapsed = ( downloadEnd - downloadStart) 
        print("Download weights round {} took {:.4} seconds".format(round_, elapsed))
        print('Imported weights in client: {} for round: {}'.format(client_, round_))

        # All neighbors has finished training
        if len(neighbors) == 0:

            print('No more neighbors in client: {}'.format(client_))
            messages = []
            for snippet in range(MODEL_SNIPPETS):
                payload = zlib.compress(cPickle.dumps({'weights': [copy.copy(WEIGHTS_CLIENTS[client_][layer]) for layer in range(cutting_points[snippet], cutting_points[snippet+1])],
                                        'snippet': snippet, 
                                        'client': client_, 
                                        'round': round_,
                                        'metrics': -1,
                                        'Stop_Training': True}))
                messages.append({'topic':client_weights_topic + '/{}/{}'.format(client_, snippet), 'payload': payload, 'qos': 2, 'retain': True})

            publish.multiple(messages, hostname=broker_address, port=MQTT_port, client_id="Client{}".format(client_), keepalive=60,
                            will=None, auth=AUTH_, tls=TLS_, protocol=mqtt.MQTTv311, transport="tcp")
            break

    ##############################################################################################################
        algorithm.update_and_assign_weights_Client(client_, round_, START_ROUND, model_client, neighbors, NUM_CLIENTS, num_images_train, num_layers, WEIGHTS_CLIENTS, GRADIENTS_CLIENTS)

        model_client = algorithm.model_client
        WEIGHTS_CLIENTS = algorithm.WEIGHTS_CLIENTS
        GRADIENTS_CLIENTS = algorithm.GRADIENTS_CLIENTS
    ##############################################################################################################

        # Validation
        validationStart = time.time()
        if round_ in VALIDATION_ROUNDS:
            history_eval = model_client.evaluate(x=valid_dataset_preprocess,
                                                steps=int(np.ceil(num_images_valid[client_] / BATCH_SIZE)),
                                                verbose=0)
            history_eval = {'accuracy': [history_eval[1:]], 'loss': [history_eval[0]]}
        validationEnd = time.time()
        valid_elapsed = ( validationEnd - validationStart) 
        print("validation round {} took {:.4} seconds".format(round_, valid_elapsed))

        # Save weights
        if save_weights:
            Save_model([copy.copy(model_client.weights[layer].numpy()) for layer in range(len(model_client.weights))], client_, cwd, round_)
            
        # Save statistics
        result.append({'round': round_, 'val_accuracy': copy.copy(history_eval['accuracy']), 'val_loss': copy.copy(history_eval['loss'])})
        np.save(cwd + f'/result_client_{client_}.npy', result, allow_pickle = True)

        # Early Stopping
        latest_losses = shift(latest_losses, -1, cval=np.NaN) 
        latest_losses[PATIENCE - 1] = history_eval['loss'][0] # Loss

        if np.argmin(latest_losses) == 0:
            Stop_Training = True

        # Train
        # print('Start training client: {} for round: {}'.format(client_, round_))
        trainStart = time.time()
        if NUM_PARTS_DATASET != 1:
            train_dataset_preprocess = dataset.create_tf_dataset_for_client_train(client_, round_)
            num_batches = min(int(np.ceil(num_images_train[client_] / BATCH_SIZE)*PERCENTAGE_DATASET), int( int( ( 1-log2(1 - round_/NUM_ROUNDS)  )) * 1/NUM_PARTS_DATASET *  int(np.ceil(num_images_train[client_] / BATCH_SIZE)*PERCENTAGE_DATASET)))
        else:
            num_batches = int(np.ceil(num_images_train[client_] / BATCH_SIZE))
        history = model_client.fit(x=train_dataset_preprocess,          
                y=None, 
                epochs=NUM_EPOCHS, 
                steps_per_epoch=num_batches,
                verbose = 1)
        trainEnd = time.time()
        elapsed = ( trainEnd - trainStart) 
        print("training round {} took {:.4} seconds".format(round_, elapsed))
        # print('Finish training client:  {} for round: {}'.format(client_, round_))

        # metrics_.append(history.history)
        # print('Client {} Round {} Accuracy: {}, Loss: {}'.format(client_, round_, history.history['accuracy'][-1], history.history['loss'][-1]))

        # UpLoad weights
        WEIGHTS_CLIENTS[client_] = [copy.copy(model_client.weights[layer].numpy()) for layer in range(num_layers)]

    ##############################################################################################################
        algorithm.set_regularizer_in_optimizer_Client(model_client, optimizer, num_layers, WEIGHTS_CLIENTS[client_])

        optimizer = algorithm.optimizer
        model_client = algorithm.model_client
    ##############################################################################################################

        messages = []
        for snippet in range(MODEL_SNIPPETS):
            payload = zlib.compress(cPickle.dumps({'weights': [copy.copy(WEIGHTS_CLIENTS[client_][layer]) for layer in range(cutting_points[snippet], cutting_points[snippet+1])],
                                    'snippet': snippet, 
                                    'client': client_, 
                                    'round': round_,
                                    'metrics': history_eval,
                                    'Stop_Training': Stop_Training,
                                    'timing': {'upload': (uploadEnd - uploadStart) if ('uploadEnd' in vars() or 'uploadEnd' in globals()) else -1, 
                                                'validation': (validationEnd - validationStart),
                                                'training': (trainEnd - trainStart),
                                                'download': (downloadEnd - downloadStart) if ('downloadEnd' in vars() or 'downloadEnd' in globals()) else -1}}))
            messages.append({'topic':client_weights_topic + '/{}/{}'.format(client_, snippet), 'payload': payload, 'qos': 2, 'retain': True})

        uploadStart = time.time()
        publish.multiple(messages, hostname=broker_address, port=MQTT_port, client_id="Client{}".format(client_), keepalive=60,
                        will=None, auth=AUTH_, tls=TLS_, protocol=mqtt.MQTTv311, transport="tcp")
        uploadEnd = time.time()
        elapsed = ( uploadEnd - uploadStart) 
        print("Upload weights round {} took {:.4} seconds".format(round_, elapsed))

        print('Published weights client: {} for round: {}'.format(client_, round_))

        # Show timing information for the round
        roundEnd = time.time()
        elapsed = ( roundEnd - roundStart) 
        print("Client {} round {} took {:.4} seconds\n".format(client_, round_, elapsed))

        # If Gossip, select new neighbor
        # neighbors = NEIGHBORS if not Gossip else np.random.choice(NEIGHBORS, size=1, replace=False)

        # If could come new Clients
        if param.INCOMING_CLIENTS and round_ > START_ROUND :
            # Check if there is a new Client
            import Classes.Params.param_mutable as param_mutable
            param_mutable.check_new_client(client_)
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
                                        PROXIMITY_MATRIX = PROXIMITY_MATRIX if PROXIMITY_MATRIX is not None else None, 
                                        WEIGHTS_CLIENTS = WEIGHTS_CLIENTS if WEIGHTS_CLIENTS is not None else None, 
                                        GRADIENTS_CLIENTS = GRADIENTS_CLIENTS if GRADIENTS_CLIENTS is not None else None, 
                                        theta_tilde = algorithm.theta_tilde if hasattr(algorithm, 'theta_tilde') else None, 
                                        DATASET = DATASET if DATASET is not None else None)

                NUM_CLIENTS = new_var.NUM_CLIENTS
                CLIENTS_SELECTED = new_var.CLIENTS_SELECTED
                WEIGHTS_CLIENTS = new_var.WEIGHTS_CLIENTS
                GRADIENTS_CLIENTS = new_var.GRADIENTS_CLIENTS
                theta_tilde = new_var.theta_tilde
                PROXIMITY_MATRIX = new_var.PROXIMITY_MATRIX
                NEIGHBORS = new_var.NEIGHBORS
                dataset = new_var.dataset
                num_images_train = new_var.num_images_train
    ##############################################################################################################

        # Early Stopping
        if Stop_Training:

            print('Early Stopping client: {} at round: {}'.format(client_, round_))
            Ping(client_, -1, {}, Last_seen_time_topic, broker_address, MQTT_port, AUTH_, TLS_, Stop_Training = 1)
            break

        if Check_stop_train(params_topic, broker_address, MQTT_port, AUTH_, TLS_):

            print('Stopping client: {} at round: {}'.format(client_, round_))
            Ping(client_, -1, {}, Last_seen_time_topic, broker_address, MQTT_port, AUTH_, TLS_, Stop_Training = 1)
            break    
        
    print('Finish client: {}'.format(client_))
