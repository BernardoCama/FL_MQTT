import copy
import numpy as np


class NewClient:
    
    def __init__(self):
        pass

    def update_variables(self, round_ = None, client_ = None, NUM_ROUNDS = None, NUM_CLIENTS = None, CLIENTS_SELECTED = None, Working_clients = None, PROXIMITY_MATRIX = None, WEIGHTS_CLIENTS = None, GRADIENTS_CLIENTS = None, partial_metrics = None, theta_tilde = None, DATASET = None):
    
        # COMMON

        # Update CLIENTS_SELECTED
        if CLIENTS_SELECTED is not None:
            CLIENTS_SELECTED = CLIENTS_SELECTED + 1
        else:
            CLIENTS_SELECTED = None

        # # Update Working_clients
        # if Working_clients is not None:
        #     Working_clients = np.array([1]*(NUM_CLIENTS))
        # else:
        #     Working_clients = None

        # Update weights
        if WEIGHTS_CLIENTS is not None:
            WEIGHTS_CLIENTS.append(copy.copy(WEIGHTS_CLIENTS[0]))
        else:
            WEIGHTS_CLIENTS = None

        # Update gradients
        if GRADIENTS_CLIENTS is not None:
            GRADIENTS_CLIENTS.append(copy.copy(GRADIENTS_CLIENTS[0]))
        else:
            GRADIENTS_CLIENTS = None
            
        # FedAdp
        if theta_tilde is not None:
            theta_tilde.append(np.mean(theta_tilde))
        else:
            theta_tilde = None

        # Update metrics
        if partial_metrics is not None:
            for i in range(NUM_ROUNDS):
                    if i < round_:
                        partial_metrics[i].append(-1)
                    else: 
                        partial_metrics[i].append(0)
        else:
            partial_metrics = None

        # Update Proximity Matrix
        if PROXIMITY_MATRIX is not None:
            PROXIMITY_MATRIX = np.ones((NUM_CLIENTS,NUM_CLIENTS))
            NEIGHBORS = np.where(PROXIMITY_MATRIX[client_, :] == 1)[0]
            NEIGHBORS = NEIGHBORS[NEIGHBORS!= client_]
        else:
            PROXIMITY_MATRIX = None
            NEIGHBORS = None

        # Update knowledge of number of samples in new Client
        if DATASET is not None:
            dataset = DATASET(NUM_CLIENTS = NUM_CLIENTS)
            num_images_train = dataset.return_num_images_train()
        else:
            dataset = None
            num_images_train = None

        self.NUM_CLIENTS = NUM_CLIENTS
        self.CLIENTS_SELECTED = CLIENTS_SELECTED
        self.Working_clients = Working_clients
        self.WEIGHTS_CLIENTS = WEIGHTS_CLIENTS
        self.GRADIENTS_CLIENTS = GRADIENTS_CLIENTS
        self.theta_tilde = theta_tilde
        self.partial_metrics = partial_metrics
        self.PROXIMITY_MATRIX = PROXIMITY_MATRIX
        self.NEIGHBORS = NEIGHBORS
        self.dataset = dataset
        self.num_images_train = num_images_train