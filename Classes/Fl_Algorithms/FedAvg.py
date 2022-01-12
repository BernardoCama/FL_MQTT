# Import parameters
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from Classes.Params.param import *

class FEDAVG:

    def __init__(self):
        pass

    """ Parameter Server Based Architecture """
    def initialize_local_variables_Server(self, model_server, num_layers, NUM_CLIENTS):

        WEIGHTS_GLOBAL = [copy.copy(model_server.weights[layer].numpy()) for layer in range(num_layers)]
        WEIGHTS_CLIENTS = [copy.copy(WEIGHTS_GLOBAL) for client in range(NUM_CLIENTS)]

        self.WEIGHTS_GLOBAL = WEIGHTS_GLOBAL
        self.WEIGHTS_CLIENTS = WEIGHTS_CLIENTS

        self.GRADIENTS_GLOBAL = None
        self.GRADIENTS_CLIENTS = None

    """ Parameter Server Based Architecture """        
    def update_weights_Server(self, round_, NUM_CLIENTS, num_images_train, sample_clients, CLIENTS_SELECTED, num_layers, WEIGHTS_GLOBAL, WEIGHTS_CLIENTS, GRADIENTS_GLOBAL, GRADIENTS_CLIENTS):
    
        # Update weights
        number_samples = np.sum([num_images_train[client_]  for  client_ in sample_clients])   
        for layer_ in range(num_layers):
            WEIGHTS_GLOBAL[layer_] = (1 - EPSILON_GLOBAL_MODEL) * WEIGHTS_GLOBAL[layer_] + \
                    EPSILON_GLOBAL_MODEL * 1/number_samples * (np.sum([WEIGHTS_CLIENTS[client_][layer_] * num_images_train[client_]  for  client_ in sample_clients], 0 ))   
        WEIGHTS_CLIENTS = [copy.copy(WEIGHTS_GLOBAL) for client in range(NUM_CLIENTS)]

        self.WEIGHTS_GLOBAL = WEIGHTS_GLOBAL
        self.WEIGHTS_CLIENTS = WEIGHTS_CLIENTS

        self.GRADIENTS_GLOBAL = None
        self.GRADIENTS_CLIENTS = None

    """ Parameter Server Based Architecture """  
    def set_learning_rate_and_assign_weights_Client(self, model_client, lr, num_layers, WEIGHTS_GLOBAL):

        tf.keras.backend.set_value(model_client.optimizer.lr, lr)  
       
        # Load weights
        for layer_ in range(num_layers):   
            model_client.weights[layer_].assign(WEIGHTS_GLOBAL[layer_])

        self.model_client = model_client

    def set_regularizer_in_optimizer_Client(self, model_client, optimizer, num_layers, WEIGHTS_GLOBAL):

        self.optimizer = optimizer
        self.model_client = model_client

    """ Consensus Based Architecture """   
    def initialize_local_variables_Client(self, model_client, num_layers, NUM_CLIENTS):

        WEIGHTS_CLIENTS = [copy.copy(model_client.weights[layer].numpy()) for layer in range(num_layers)]
        WEIGHTS_CLIENTS = [copy.copy(WEIGHTS_CLIENTS) for client in range(NUM_CLIENTS)]

        self.WEIGHTS_CLIENTS = WEIGHTS_CLIENTS
        self.GRADIENTS_CLIENTS = None

    """ Consensus Based Architecture """        
    def update_and_assign_weights_Client(self, client_, round_, START_ROUND, model_client, neighbors, NUM_CLIENTS, num_images_train, num_layers, WEIGHTS_CLIENTS, GRADIENTS_CLIENTS):
        
        # Update weights
        number_samples = np.sum([num_images_train[neighbor]  for  neighbor in neighbors])
        for layer_ in range(num_layers):

            WEIGHTS_CLIENTS[client_][layer_] =  WEIGHTS_CLIENTS[client_][layer_] + \
                                            EPSILON *  1/number_samples * \
                                            (np.sum([ num_images_train[neighbor] * (WEIGHTS_CLIENTS[neighbor][layer_] - WEIGHTS_CLIENTS[client_][layer_])  for  neighbor in  neighbors], 0 ))    
            model_client.weights[layer_].assign(WEIGHTS_CLIENTS[client_][layer_])

        self.model_client = model_client
        self.WEIGHTS_CLIENTS = WEIGHTS_CLIENTS
        self.GRADIENTS_CLIENTS = None






