# MacOs
# import os
# import plaidml.keras
# plaidml.keras.install_backend()
# os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
# import keras

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from Classes.Params import param
import tensorflow as tf


class VGG16:
    
    def __init__(self, input_shape):

        self.input_shape = input_shape

    # Model definition
    def return_model(self):

        vgg = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)

        finetuning = False

        if finetuning:
            freeze_until = 15 # layer from which we want to fine-tune
            
            for layer in vgg.layers[:freeze_until]:
                layer.trainable = False
        else:
            vgg.trainable = False
            
        model = tf.keras.Sequential()
        model.add(vgg)
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=16, activation='relu')) #512
        model.add(tf.keras.layers.Dense(units=param.num_classes, activation='softmax'))

        #model.summary()
        
        return model