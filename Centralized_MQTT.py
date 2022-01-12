#!/usr/bin/env python
# coding: utf-8

# Client counterpart in FL with Consensus (NO TFF)

# Import parameters
from Classes.Params.param import *
from Classes.Auxiliary.Cut_model import Cut_model
from Classes.Auxiliary.Save_model import Save_model
param.time.sleep(3)
from keras.callbacks import CSVLogger

# ID of the client
client_ = 0

# Import dataset
DATASET = getattr(importlib.import_module(MODULE_DATASET_NAME), DATASET_NAME)

# Initialize dataset
dataset = DATASET()

num_images_train = dataset.return_num_images_train()
num_images_test = dataset.return_num_images_test()
num_images_valid = dataset.return_num_images_valid()
input_shape = dataset.return_input_shape()
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
    size_model = sys.getsizeof((cPickle.dumps(WEIGHTS_CLIENTS)))
    print('Model memory size uncompressed in B: {}, KB: {}, MB: {}'.format(size_model, size_model/10**3, size_model/10**6))
    size_model = sys.getsizeof(zlib.compress(cPickle.dumps(WEIGHTS_CLIENTS)))
    print('Model memory size compressed in B: {}, KB: {}, MB: {}'.format(size_model, size_model/10**3, size_model/10**6))
    for layer in range(num_layers):
        print('layer: {}, uncompressed: {} KB, compressed: {} KB'.format(layer, 
                        sys.getsizeof(cPickle.dumps(WEIGHTS_CLIENTS[layer]))/10**3,
                        sys.getsizeof(zlib.compress(cPickle.dumps(WEIGHTS_CLIENTS[layer])))/10**3))
    
    cutting_points, MODEL_SNIPPETS = Cut_model(size_model, MAX_MODEL_SIZE, num_layers, WEIGHTS_CLIENTS)

    # Take datasets
    test_dataset_preprocess = dataset.create_tf_dataset_for_client_test(client_)
    valid_dataset_preprocess = dataset.create_tf_dataset_for_client_valid(client_)
    if NUM_PARTS_DATASET != 1:
        train_dataset_preprocess = dataset.create_tf_dataset_for_client_train(client_, 0)
    else:
        train_dataset_preprocess = dataset.create_tf_dataset_for_client_train(client_)

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

    # Callbacks
    callbacks = []
    ckpt_dir = cwd + '/H{}'.format(client_) + '/saved_weights'

    exp_dir = os.path.join(ckpt_dir)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    if save_weights:    
    # Model checkpoint
        ckpt_dir = os.path.join(exp_dir, 'ckpts')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp_{epoch:02d}.ckpt'), 
                                                        save_weights_only=False)  # False to save the model directly
        callbacks.append(ckpt_callback)

        # csv_logger = CSVLogger(os.path.join(ckpt_dir, "model_history_log_{epoch:02d}.csv") , append=True)
        # callbacks.append(csv_logger)


    # Early Stopping
    # --------------
    early_stop = True
    if early_stop:
        es_callback = tf.keras.callbacks.EarlyStopping(monitor= 'val_loss' , patience=PATIENCE)
        callbacks.append(es_callback)

    # Training
    trainStart = time.time()
    history = model_client.fit(x=train_dataset_preprocess,          
                y=None, 
                epochs=NUM_ROUNDS, 
                steps_per_epoch=int(np.ceil(num_images_train[client_] / BATCH_SIZE)),
                validation_data=valid_dataset_preprocess,  
                validation_steps=int(np.ceil(num_images_valid[client_] / BATCH_SIZE)),
                callbacks=callbacks,
                verbose = 1)
    trainEnd = time.time()
    elapsed = ( trainEnd - trainStart) 
    print(f"training took {elapsed:.4f} seconds")

    history = history.history

    result = []
    key = list(history.keys())
    len_metric = len(key)

    NUM_ROUNDS = len( history[key[0]])

    # Prepare results
    for round_ in range(NUM_ROUNDS):
        result.append({'round': round_, 'time_round': elapsed/NUM_ROUNDS, 
        'mean_accuracy': [np.array([ history[key[int(len_metric/2) + i + 1]][round_]  for i in range(int(len_metric/2) - 1)])], 
        'mean_loss': history[key[int(len_metric/2)]][round_],
        'mean_accuracy_train':  [np.array([ history[key[i + 1]][round_]  for i in range(int(len_metric/2) - 1)])], 
        'mean_loss_train':  history[key[0]][round_]
        })

    np.save(cwd + '/result.npy', copy.copy(result), allow_pickle = True)
    # result = np.load(cwd + '/result.npy', allow_pickle = True)
    # print(result)  
    print('Finish client: {}'.format(client_))
