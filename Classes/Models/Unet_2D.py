import tensorflow as tf

# MacOs
# import os
# import plaidml.keras
# plaidml.keras.install_backend()
# os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
# import keras

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from Classes.Params import param


class UNET_2D:
    
    def __init__(self, input_shape):

        self.input_shape = input_shape


    # Model definition
    def return_model(self):

        imgs_shape = self.input_shape

        msks_shape = [self.input_shape[0], self.input_shape[1], param.num_classes]

        use_dropout = 1 #False

        dropout = 0.2   

        use_upsampling = 1 #False

        start_f = 32

        inputs = tf.keras.layers.Input(imgs_shape, name="MRImages")

        # Convolution parameters
        params = dict(kernel_size=(3, 3), activation="relu",
                    padding="same",
                    kernel_initializer="he_uniform")

        # Transposed convolution parameters
        params_trans = dict(kernel_size=(2, 2), strides=(2, 2),
                            padding="same")

        encodeA = tf.keras.layers.Conv2D(
            name="encodeAa", filters=start_f, **params)(inputs)
        encodeA = tf.keras.layers.Conv2D(
            name="encodeAb", filters=start_f, **params)(encodeA)
        poolA = tf.keras.layers.MaxPooling2D(name="poolA", pool_size=(2, 2))(encodeA)

        encodeB = tf.keras.layers.Conv2D(
            name="encodeBa", filters=start_f*2, **params)(poolA)
        encodeB = tf.keras.layers.Conv2D(
            name="encodeBb", filters=start_f*2, **params)(encodeB)
        poolB = tf.keras.layers.MaxPooling2D(name="poolB", pool_size=(2, 2))(encodeB)

        encodeC = tf.keras.layers.Conv2D(
            name="encodeCa", filters=start_f*4, **params)(poolB)
        if use_dropout:
            encodeC = tf.keras.layers.SpatialDropout2D(dropout)(encodeC)
        encodeC = tf.keras.layers.Conv2D(
            name="encodeCb", filters=start_f*4, **params)(encodeC)

        poolC = tf.keras.layers.MaxPooling2D(name="poolC", pool_size=(2, 2))(encodeC)

        encodeD = tf.keras.layers.Conv2D(
            name="encodeDa", filters=start_f*8, **params)(poolC)
        if use_dropout:
            encodeD = tf.keras.layers.SpatialDropout2D(dropout)(encodeD)
        encodeD = tf.keras.layers.Conv2D(
            name="encodeDb", filters=start_f*8, **params)(encodeD)

        poolD = tf.keras.layers.MaxPooling2D(name="poolD", pool_size=(2, 2))(encodeD)

        encodeE = tf.keras.layers.Conv2D(
            name="encodeEa", filters=start_f*16, **params)(poolD)
        encodeE = tf.keras.layers.Conv2D(
            name="encodeEb", filters=start_f*16, **params)(encodeE)

        if use_upsampling:
            up = tf.keras.layers.UpSampling2D(name="upE", size=(2, 2))(encodeE)
        else:
            up = tf.keras.layers.Conv2DTranspose(name="transconvE", filters=start_f*8,
                                        **params_trans)(encodeE)
        concatD = tf.keras.layers.concatenate(
            [up, encodeD], axis=-1, name="concatD")

        decodeC = tf.keras.layers.Conv2D(
            name="decodeCa", filters=start_f*8, **params)(concatD)
        decodeC = tf.keras.layers.Conv2D(
            name="decodeCb", filters=start_f*8, **params)(decodeC)

        if use_upsampling:
            up = tf.keras.layers.UpSampling2D(name="upC", size=(2, 2))(decodeC)
        else:
            up = tf.keras.layers.Conv2DTranspose(name="transconvC", filters=start_f*4,
                                        **params_trans)(decodeC)
        concatC = tf.keras.layers.concatenate(
            [up, encodeC], axis=-1, name="concatC")

        decodeB = tf.keras.layers.Conv2D(
            name="decodeBa", filters=start_f*4, **params)(concatC)
        decodeB = tf.keras.layers.Conv2D(
            name="decodeBb", filters=start_f*4, **params)(decodeB)

        if use_upsampling:
            up = tf.keras.layers.UpSampling2D(name="upB", size=(2, 2))(decodeB)
        else:
            up = tf.keras.layers.Conv2DTranspose(name="transconvB", filters=start_f*2,
                                        **params_trans)(decodeB)
        concatB = tf.keras.layers.concatenate(
            [up, encodeB], axis=-1, name="concatB")

        decodeA = tf.keras.layers.Conv2D(
            name="decodeAa", filters=start_f*2, **params)(concatB)
        decodeA = tf.keras.layers.Conv2D(
            name="decodeAb", filters=start_f*2, **params)(decodeA)

        if use_upsampling:
            up = tf.keras.layers.UpSampling2D(name="upA", size=(2, 2))(decodeA)
        else:
            up = tf.keras.layers.Conv2DTranspose(name="transconvA", filters=start_f,
                                        **params_trans)(decodeA)
        concatA = tf.keras.layers.concatenate(
            [up, encodeA], axis=-1, name="concatA")

        convOut = tf.keras.layers.Conv2D(
            name="convOuta", filters=start_f, **params)(concatA)
        convOut = tf.keras.layers.Conv2D(
            name="convOutb", filters=start_f, **params)(convOut)

        prediction = tf.keras.layers.Conv2D(name="PredictionMask",
                                    filters=param.num_classes, kernel_size=(1, 1),
                                    activation="softmax")(convOut)

        model = tf.keras.models.Model(inputs=[inputs], outputs=[
                            prediction], name="2DUNet_Brats")

        if param.CONTINUAL_LEARNING:
            new_weights = param.np.load(param.MODEL_WEIGHTS_FILE, allow_pickle = True)
            # Load weights
            for layer_ in range(len(model.weights)):   
                model.weights[layer_].assign(new_weights[layer_])      
                
        # model.summary() 
        
        return model