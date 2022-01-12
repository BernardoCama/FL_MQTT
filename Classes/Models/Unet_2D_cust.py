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


class UNET_2D_cust:
    
    def __init__(self, input_shape):

        self.input_shape = input_shape

    def unet(self,inputs, nb_classes=param.num_classes, start_ch=64, depth=3, inc_rate=2. ,activation='relu', dropout=0.0, batchnorm=True, upconv=True,format_='channels_last'):
        """
        the actual u-net architecture
        """
        o = self.level_block(inputs,start_ch, depth, inc_rate,activation, dropout, batchnorm, upconv,format_)
        o = tf.keras.layers.BatchNormalization()(o) 
        #o =  Activation('relu')(o)
        o = tf.keras.layers.PReLU(shared_axes=[1, 2])(o)
        o = tf.keras.layers.Conv2D(nb_classes, 1, padding='same',data_format = format_)(o)
        o = tf.keras.layers.Activation('softmax')(o)
        return o

    def level_block(self,m, dim, depth, inc, acti, do, bn, up,format_="channels_last"):
        if depth > 0:
            n = self.res_block_enc(m,0.0,dim,acti, bn,format_)
            #using strided 2D conv for donwsampling
            m = tf.keras.layers.Conv2D(int(inc*dim), 2,strides=2, padding='same',data_format = format_)(n)
            m = self.level_block(m,int(inc*dim), depth-1, inc, acti, do, bn, up )
            if up:
                m = tf.keras.layers.UpSampling2D(size=(2, 2),data_format = format_)(m)
                m = tf.keras.layers.Conv2D(dim, 2, padding='same',data_format = format_)(m)
            else:
                m = tf.keras.layers.Conv2DTranspose(dim, 3, strides=2,padding='same',data_format = format_)(m)
            n = tf.keras.layers.concatenate([n,m])
            #the decoding path
            m = self.res_block_dec(n, 0.0,dim, acti, bn, format_)
        else:
            m = self.res_block_enc(m, 0.0,dim, acti, bn, format_)
        return m

    def res_block_enc(self,m, drpout,dim,acti, bn,format_="channels_last"):
        
        """
        the encoding unit which a residual block
        """
        n = tf.keras.layers.BatchNormalization()(m) if bn else n
        #n=  Activation(acti)(n)
        n = tf.keras.layers.PReLU(shared_axes=[1, 2])(n)
        n = tf.keras.layers.Conv2D(dim, 3, padding='same',data_format = format_)(n)
                
        n = tf.keras.layers.BatchNormalization()(n) if bn else n
        #n=  Activation(acti)(n)
        n = tf.keras.layers.PReLU(shared_axes=[1, 2])(n)
        n = tf.keras.layers.Conv2D(dim, 3, padding='same',data_format =format_ )(n)

        n = tf.keras.layers.add([m,n]) 
        
        return  n 

    def res_block_dec(self,m, drpout,dim,acti, bn,format_="channels_last"):

        """
        the decoding unit which a residual block
        """
         
        n = tf.keras.layers.BatchNormalization()(m) if bn else n
        #n=  Activation(acti)(n)
        n = tf.keras.layers.PReLU(shared_axes=[1, 2])(n)
        n = tf.keras.layers.Conv2D(dim, 3, padding='same',data_format = format_)(n)

        n = tf.keras.layers.BatchNormalization()(n) if bn else n
        #n=  Activation(acti)(n)
        n = tf.keras.layers.PReLU(shared_axes=[1, 2])(n)
        n = tf.keras.layers.Conv2D(dim, 3, padding='same',data_format =format_ )(n)
        
        Save = tf.keras.layers.Conv2D(dim, 1, padding='same',data_format = format_,use_bias=False)(m) 
        n = tf.keras.layers.add([Save,n]) 
        
        return  n   


    # Model definition
    def return_model(self):

        imgs_shape = self.input_shape

        msks_shape = [self.input_shape[0], self.input_shape[1], param.num_classes]

        i = tf.keras.layers.Input(shape=imgs_shape)
        #add gaussian noise to the first layer to combat overfitting
        i_= tf.keras.layers.GaussianNoise(0.01)(i)

        i_ = tf.keras.layers.Conv2D(64, 2, padding='same',data_format = 'channels_last')(i_)
        out=self.unet(inputs=i_)
        model = tf.keras.models.Model(inputs=i, outputs=out)

        if param.CONTINUAL_LEARNING:
            new_weights = param.np.load(param.MODEL_WEIGHTS_FILE, allow_pickle = True)
            # Load weights
            for layer_ in range(len(model.weights)):   
                model.weights[layer_].assign(new_weights[layer_])      
                
        print(model.summary())
        
        return model