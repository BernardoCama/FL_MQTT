# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D,\
     Flatten, BatchNormalization, AveragePooling2D, Dense, Activation, Add 
from tensorflow.keras.models import Model
from tensorflow.keras import activations
from tensorflow.keras.regularizers import l2

# MacOs
# import os
# import plaidml.keras
# plaidml.keras.install_backend()
# os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
# import keras

from Classes.Params import param


class RESNET50:
    
    def __init__(self, input_shape):

        self.input_shape = input_shape
            
    def res_identity(self, x, filters): 
        ''' renet block where dimension doesnot change.
        The skip connection is just simple identity conncection
        we will have 3 blocks and then input will be added
        '''
        x_skip = x # this will be used for addition with the residual block 
        f1, f2 = filters

        #first block 
        x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)

        #second block # bottleneck (but size kept same with padding)
        x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)

        # third block activation used after adding the input
        x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        # x = Activation(activations.relu)(x)

        # add the input 
        x = Add()([x, x_skip])
        x = Activation(activations.relu)(x)

        return x

    def res_conv(self, x, s, filters):
        '''
        here the input size changes, when it goes via conv blocks
        so the skip connection uses a projection (conv layer) matrix
        ''' 
        x_skip = x
        f1, f2 = filters

        # first block
        x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x)
        # when s = 2 then it is like downsizing the feature map
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)

        # second block
        x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)

        #third block
        x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)

        # shortcut 
        x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x_skip)
        x_skip = BatchNormalization()(x_skip)

        # add 
        x = Add()([x, x_skip])
        x = Activation(activations.relu)(x)

        return x


    # Model definition
    def return_model(self):
    
        input_im = Input(shape=self.input_shape)
        x = ZeroPadding2D(padding=(3, 3))(input_im)

        # 1st stage
        # here we perform maxpooling, see the figure above

        x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        #2nd stage 
        # frm here on only conv block and identity block, no pooling

        x = self.res_conv(x, s=1, filters=(64, 256))
        x = self.res_identity(x, filters=(64, 256))
        x = self.res_identity(x, filters=(64, 256))

        # 3rd stage

        x = self.res_conv(x, s=2, filters=(128, 512))
        x = self.res_identity(x, filters=(128, 512))
        x = self.res_identity(x, filters=(128, 512))
        x = self.res_identity(x, filters=(128, 512))

        # 4th stage

        x = self.res_conv(x, s=2, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))

        # 5th stage

        x = self.res_conv(x, s=2, filters=(512, 2048))
        x = self.res_identity(x, filters=(512, 2048))
        x = self.res_identity(x, filters=(512, 2048))

        # ends with average pooling and dense connection

        x = AveragePooling2D((2, 2), padding='same')(x)

        x = Flatten()(x)
        x = Dense(param.num_classes, activation='softmax', kernel_initializer='he_normal')(x) #multi-class

        # define the model 

        model = Model(inputs=input_im, outputs=x, name='Resnet50')

        #model.summary()

        return model

