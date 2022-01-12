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
class UNET_2D_cust2(object):
    
    def __init__(self,input_shape):
        self.input_shape = input_shape
      

    def custom_unet(self,
        inputs,
        num_classes=param.num_classes,
        activation="relu",
       # activation=tf.keras.layers.LeakyReLU(alpha=0.1),
        use_batch_norm=True,
        upsample_mode="deconv",  # 'deconv' or 'simple'
        dropout=0.0,
        dropout_change_per_layer=0.0,
        dropout_type="spatial",
        use_dropout_on_upsampling=False,
        filters=32,
        num_layers=4,
        output_activation="softmax",
    ):  # 'sigmoid' or 'softmax'
    
        """
        Customisable UNet architecture (Ronneberger et al. 2015 [1]).
    
        Arguments:
        input_shape: 3D Tensor of shape (x, y, num_channels)
    
        num_classes (int): Unique classes in the output mask. Should be set to 1 for binary segmentation
    
        activation (str): A keras.activations.Activation to use. ReLu by default.
    
        use_batch_norm (bool): Whether to use Batch Normalisation across the channel axis between convolutional layers
    
        upsample_mode (one of "deconv" or "simple"): Whether to use transposed convolutions or simple upsampling in the decoder part
    
        dropout (float between 0. and 1.): Amount of dropout after the initial convolutional block. Set to 0. to turn Dropout off
    
        dropout_change_per_layer (float between 0. and 1.): Factor to add to the Dropout after each convolutional block
    
        dropout_type (one of "spatial" or "standard"): Type of Dropout to apply. Spatial is recommended for CNNs [2]
    
        use_dropout_on_upsampling (bool): Whether to use dropout in the decoder part of the network
    
        filters (int): Convolutional filters in the initial convolutional block. Will be doubled every block
    
        num_layers (int): Number of total layers in the encoder not including the bottleneck layer
    
        output_activation (str): A keras.activations.Activation to use. Sigmoid by default for binary segmentation
    
        Returns:
        model (keras.models.Model): The built U-Net
    
        Raises:
        ValueError: If dropout_type is not one of "spatial" or "standard"
        """
    
        if upsample_mode == "deconv":
            upsample = self.upsample_conv
        else:
            upsample = self.upsample_simple
    
        # Build U-Net model
        #inputs = Input(input_shape)
        x = inputs
    
        down_layers = []
        for l in range(num_layers):
            x = self.conv2d_block(
                inputs=x,
                filters=filters,
                use_batch_norm=use_batch_norm,
                dropout=dropout,
                dropout_type=dropout_type,
                activation=activation,
            )
            down_layers.append(x)
            x = tf.keras.layers.MaxPooling2D((2, 2))(x)
            dropout += dropout_change_per_layer
            filters = filters * 2  # double the number of filters with each layer
    
        x = self.conv2d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            dropout_type=dropout_type,
            activation=activation,
        )
    
        if not use_dropout_on_upsampling:
            dropout = 0.0
            dropout_change_per_layer = 0.0
    
        for conv in reversed(down_layers):
            filters //= 2  # decreasing number of filters with each layer
            dropout -= dropout_change_per_layer
            x = upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
            x = tf.keras.layers.concatenate([x, conv])
            x = self.conv2d_block(
                inputs=x,
                filters=filters,
                use_batch_norm=use_batch_norm,
                dropout=dropout,
                dropout_type=dropout_type,
                activation=activation,
            )
    
        outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation=output_activation)(x)
    
        #model = Model(inputs=[inputs], outputs=[outputs])
        return outputs    
    


        
    def upsample_conv(self,filters, kernel_size, strides, padding):
        return tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)
    
    
    def upsample_simple(self,filters, kernel_size, strides, padding):
        return tf.keras.layers.UpSampling2D(strides)
    
    
    def conv2d_block(self,
        inputs,
        use_batch_norm=True,
        dropout=0.3,
        dropout_type="spatial",
        filters=16,
        kernel_size=(3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    ):
    
        if dropout_type == "spatial":
            DO = tf.keras.layers.SpatialDropout2D
        elif dropout_type == "standard":
            DO = tf.keras.layers.Dropout
        else:
            raise ValueError(
                f"dropout_type must be one of ['spatial', 'standard'], got {dropout_type}"
            )
    
        c = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            activation=activation,
            kernel_initializer=kernel_initializer,
            padding=padding,
            use_bias=not use_batch_norm,
        )(inputs)
        if use_batch_norm:
            c = tf.keras.layers.BatchNormalization()(c)
        if dropout > 0.0:
            c = DO(dropout)(c)
        c = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            activation=activation,
            kernel_initializer=kernel_initializer,
            padding=padding,
            use_bias=not use_batch_norm,
        )(c)
        if use_batch_norm:
            c = tf.keras.layers.BatchNormalization()(c)
        return c
    
    


    # Model definition
    def return_model(self):

        imgs_shape = self.input_shape

        msks_shape = [self.input_shape[0], self.input_shape[1], param.num_classes]

        i =  tf.keras.layers.Input(shape=self.input_shape)

        out=self.custom_unet(inputs=i) 
        model = tf.keras.models.Model(inputs=i, outputs=out)

        if param.CONTINUAL_LEARNING:
            new_weights = param.np.load(param.MODEL_WEIGHTS_FILE, allow_pickle = True)
            # Load weights
            for layer_ in range(len(model.weights)):   
                model.weights[layer_].assign(new_weights[layer_])      
                
        print(model.summary())
        
        return model