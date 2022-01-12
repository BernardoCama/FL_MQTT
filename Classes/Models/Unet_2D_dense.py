import tensorflow as tf
from keras.models import Model,load_model
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout,GaussianNoise, Input,Activation
from keras.layers.normalization import BatchNormalization
from keras.layers import  Conv2DTranspose,UpSampling2D,concatenate,add
import tensorflow.keras.backend as K

K.set_image_data_format("channels_last")
from Classes.Params import param


 #Dense u-net model
class UNET_2D_dense(object):
    
    def __init__(self,input_shape):
        self.input_shape = input_shape
        
    def get_unet(self,inputs):
        #inputs = Input((img_rows, img_cols, 1))
        conv11 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conc11 = tf.keras.layers.concatenate([inputs, conv11], axis=3)
        conv12 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conc11)
        conc12 = tf.keras.layers.concatenate([inputs, conv12], axis=3)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conc12)
    
        conv21 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conc21 = tf.keras.layers.concatenate([pool1, conv21], axis=3)
        conv22 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conc21)
        conc22 = tf.keras.layers.concatenate([pool1, conv22], axis=3)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conc22)
    
        conv31 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conc31 = tf.keras.layers.concatenate([pool2, conv31], axis=3)
        conv32 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conc31)
        conc32 = tf.keras.layers.concatenate([pool2, conv32], axis=3)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conc32)
    
        conv41 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conc41 = tf.keras.layers.concatenate([pool3, conv41], axis=3)
        conv42 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conc41)
        conc42 = tf.keras.layers.concatenate([pool3, conv42], axis=3)
        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conc42)
    
        conv51 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conc51 = tf.keras.layers.concatenate([pool4, conv51], axis=3)
        conv52 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conc51)
        conc52 = tf.keras.layers.concatenate([pool4, conv52], axis=3)
    
        up6 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conc52), conc42], axis=3)
        conv61 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conc61 = tf.keras.layers.concatenate([up6, conv61], axis=3)
        conv62 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conc61)
        conc62 = tf.keras.layers.concatenate([up6, conv62], axis=3)
    
    
        up7 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conc62), conv32], axis=3)
        conv71 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conc71 = tf.keras.layers.concatenate([up7, conv71], axis=3)
        conv72 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conc71)
        conc72 = tf.keras.layers.concatenate([up7, conv72], axis=3)
    
        up8 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conc72), conv22], axis=3)
        conv81 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conc81 = tf.keras.layers.concatenate([up8, conv81], axis=3)
        conv82 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conc81)
        conc82 = tf.keras.layers.concatenate([up8, conv82], axis=3)
    
        up9 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conc82), conv12], axis=3)
        conv91 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conc91 = tf.keras.layers.concatenate([up9, conv91], axis=3)
        conv92 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conc91)
        conc92 = tf.keras.layers.concatenate([up9, conv92], axis=3)
    
        conv10 = tf.keras.layers.Conv2D(param.num_classes, (1, 1), activation='softmax')(conc92)
    
        # model = Model(inputs=[inputs], outputs=[conv10])
    
        # model.summary()
        # plot_model(model, to_file='model.png')
    
        # model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss='binary_crossentropy', metrics=['accuracy'])
    
        return conv10

    # Model definition
    def return_model(self):

        imgs_shape = self.input_shape

        msks_shape = [self.input_shape[0], self.input_shape[1], param.num_classes]

        i = tf.keras.layers.Input(shape=self.input_shape)

        # add gaussian noise to the first layer to combat overfitting
        # i_=GaussianNoise(0.01)(i)
        # i_ = tf.keras.layers.Conv2D(64, 2, padding='same',data_format = 'channels_last')(i)

        out=self.get_unet(inputs=i)
        model = tf.keras.models.Model(inputs=i, outputs=out)

        if param.CONTINUAL_LEARNING:
            new_weights = param.np.load(param.MODEL_WEIGHTS_FILE, allow_pickle = True)
            # Load weights
            for layer_ in range(len(model.weights)):   
                model.weights[layer_].assign(new_weights[layer_])      

        print(model.summary())
        
        return model