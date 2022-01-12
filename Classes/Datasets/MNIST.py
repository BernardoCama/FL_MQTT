import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
from Classes.Params import param
tf.random.set_seed(param.SEED) 
np.random.seed(param.SEED) 


class MNIST:

    def __init__(self, NUM_CLIENTS = param.NUM_CLIENTS,
                num_classes = param.num_classes,
                SHUFFLE_BUFFER = param.SHUFFLE_BUFFER,
                PREFETCH_BUFFER = param.PREFETCH_BUFFER,
                BATCH_SIZE = param.BATCH_SIZE):

        self.NUM_CLIENTS = NUM_CLIENTS
        self.num_classes = num_classes
        self.SHUFFLE_BUFFER = SHUFFLE_BUFFER
        self.PREFETCH_BUFFER = PREFETCH_BUFFER
        self.BATCH_SIZE = BATCH_SIZE
        
        # Preparation of input data
        (images, labels), (images_test, labels_test) = tf.keras.datasets.mnist.load_data()

        num_images_train = 40000
        num_images_test = 10000
        num_images_valid = 10000
        self.height = images.shape[1]
        self.width = images.shape[2]
        self.levels = 1

        # Split in training and validation sets 
        self.x_train = images[:num_images_train,...]
        self.y_train = labels[:num_images_train,...]
        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)
        self.x_train = np.array_split(self.x_train, self.NUM_CLIENTS)
        self.y_train = np.array_split(self.y_train, self.NUM_CLIENTS)

        self.x_valid = images[num_images_train:num_images_train + num_images_valid, ...] 
        self.y_valid = labels[num_images_train:num_images_train + num_images_valid, ...] 
        self.x_valid = np.array(self.x_valid)
        self.y_valid = np.array(self.y_valid)
        self.x_valid = np.array_split(self.x_valid, self.NUM_CLIENTS)
        self.y_valid = np.array_split(self.y_valid, self.NUM_CLIENTS)

        self.x_test = images_test
        self.y_test = labels_test
        self.x_test = np.array(self.x_test)
        self.y_test = np.array(self.y_test)
        self.x_test = np.array_split(self.x_test, self.NUM_CLIENTS)
        self.y_test = np.array_split(self.y_test, self.NUM_CLIENTS)

        # print ("Num images for training: {}".format(num_images_train))
        self.num_images_train = []
        for client_ in range(self.NUM_CLIENTS):
            number_samples = len(self.x_train[client_])
            self.num_images_train.append(number_samples)
            # print('Number training samples for H{}: '.format(client_) + str(number_samples))
            
        # print ("Num images for validation: {}".format(num_images_valid))
        self.num_images_valid = []
        for client_ in range(self.NUM_CLIENTS):
            number_samples = len(self.x_valid[client_])
            self.num_images_valid.append(number_samples)
            # print('Number validation samples for H{}: '.format(client_) + str(number_samples))
            
        # print ("Num images for testing: {}".format(num_images_test))
        self.num_images_test = []
        for client_ in range(self.NUM_CLIENTS):
            number_samples = len(self.x_test[client_])
            self.num_images_test.append(number_samples)
            # print('Number testing samples for H{}: '.format(client_) + str(number_samples))

    # Number images
    def return_num_images_train(self):
        return self.num_images_train
    
    def return_num_images_valid(self):
        return self.num_images_valid
    
    def return_num_images_test(self):
        return self.num_images_test
    
    # Input shape
    def return_input_shape(self):
        return [self.height, self.width, self.levels]

    # Preprocess
    def preprocess(self, dataset):
        
        # 1-hot encoding <- for categorical cross entropy
        def to_categorical(x_, y_):
                return x_, tf.one_hot(tf.cast(y_, tf.uint8), depth= self.num_classes) # depth is the numbers of classes

        # Normalize images
        def normalize_img(x_, y_):
                return tf.cast(x_, tf.float32) / 255., tf.cast(y_, tf.float32)
            
        def batch_format_fn(image, label):

                return tf.reshape(image, [-1, self.height, self.width, self.levels]), tf.reshape(label, [-1,  self.num_classes])
        
        #return dataset.repeat().batch(BATCH_SIZE).map(to_categorical).map(normalize_img).map(batch_format_fn).prefetch(PREFETCH_BUFFER)
        return dataset.repeat().shuffle(self.SHUFFLE_BUFFER).batch(self.BATCH_SIZE).map(to_categorical).map(batch_format_fn).prefetch(self.PREFETCH_BUFFER)


    def create_tf_dataset_for_client_train(self, client_id):
        return  self.preprocess(tf.data.Dataset.from_tensor_slices((self.x_train[client_id], self.y_train[client_id])))

    def create_tf_dataset_for_client_valid(self, client_id):
        return  self.preprocess(tf.data.Dataset.from_tensor_slices((self.x_valid[client_id], self.y_valid[client_id])))

    def create_tf_dataset_for_client_test(self, client_id):
        return  self.preprocess(tf.data.Dataset.from_tensor_slices((self.x_test[client_id], self.y_test[client_id])))

    # For centralized dataset
    def create_tf_dataset_train(self):
        return  self.preprocess(tf.data.Dataset.from_tensor_slices((np.concatenate([x for x in self.x_train]), np.concatenate([y for y in self.y_train]))))

    def create_tf_dataset_valid(self):
        return  self.preprocess(tf.data.Dataset.from_tensor_slices((np.concatenate([x for x in self.x_valid]), np.concatenate([y for y in self.y_valid]))))

    def create_tf_dataset_test(self):
        return  self.preprocess(tf.data.Dataset.from_tensor_slices((np.concatenate([x for x in self.x_test]), np.concatenate([y for y in self.y_test]))))

    # Preprocess
    def preprocess2(self, dataset):
        
        # 1-hot encoding <- for categorical cross entropy
        def to_categorical(x_, y_):
                return x_, tf.one_hot(tf.cast(y_, tf.uint8), depth= self.num_classes) # depth is the numbers of classes

        # Normalize images
        def normalize_img(x_, y_):
                return tf.cast(x_, tf.float32) / 255., tf.cast(y_, tf.float32)
            
        def batch_format_fn(image, label):

                return tf.reshape(image, [-1, self.height, self.width, self.levels]), tf.reshape(label, [-1,  self.num_classes])
        
        #return dataset.repeat().batch(BATCH_SIZE).map(to_categorical).map(normalize_img).map(batch_format_fn).prefetch(PREFETCH_BUFFER)
        return dataset.repeat().shuffle(self.SHUFFLE_BUFFER).batch(40000).map(to_categorical).map(batch_format_fn).prefetch(self.PREFETCH_BUFFER)

    # For CFA-GE
    def create_tf_dataset_train_unbatched(self, client_id):
        return  self.preprocess2(tf.data.Dataset.from_tensor_slices((self.x_train[client_id], self.y_train[client_id])))
        # return  self.preprocess((np.concatenate([tf.cast(x, tf.float32) / 255. for x in self.x_train]), np.concatenate([tf.one_hot(tf.cast(y, tf.uint8), depth= self.num_classes) for y in self.y_train])))

    def create_tf_dataset_valid_unbatched(self, client_id):
        return  self.preprocess2(tf.data.Dataset.from_tensor_slices((self.x_valid[client_id], self.y_valid[client_id])))
        # return  self.preprocess((np.concatenate([x for x in self.x_valid]), np.concatenate([y for y in self.y_valid])))

    def create_tf_dataset_test_unbatched(self, client_id):
        return  self.preprocess2(tf.data.Dataset.from_tensor_slices((self.x_test[client_id], self.y_test[client_id])))
        # return  self.preprocess((np.concatenate([x for x in self.x_test]), np.concatenate([y for y in self.y_test]))) 