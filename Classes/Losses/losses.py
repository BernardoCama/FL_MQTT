import numpy as np
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow.keras.backend as K
import tensorflow as tf
from Classes.Params import param

def generalized_dice(y_true, y_pred):
    
    """
    Generalized Dice Score
    https://arxiv.org/pdf/1707.03237
    
    """
    
    y_true    = K.reshape(y_true,shape=(-1,param.num_classes))
    y_pred    = K.reshape(y_pred,shape=(-1,param.num_classes))
    sum_p     = K.sum(y_pred, -2)
    sum_r     = K.sum(y_true, -2)
    sum_pr    = K.sum(y_true * y_pred, -2)
    weights   = K.pow(K.square(sum_r) + K.epsilon(), -1)
    generalized_dice = (2 * K.sum(weights * sum_pr)) / (K.sum(weights * (sum_r + sum_p)))
    
    return generalized_dice

def generalized_dice_loss(y_true, y_pred):   
    return 1-generalized_dice(y_true, y_pred)

def custom_loss(y_true, y_pred):
    
    """
    The final loss function consists of the summation of two losses "GDL" and "CE"
    with a regularization term.
    """
    
    return (0.85)*generalized_dice_loss(y_true, y_pred) + (1-0.85) * tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    # return generalized_dice_loss(y_true, y_pred) + (1.25) * tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    # return tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    

#######################################################################################################################################

def dice_coef_loss(target, prediction, axis=(1, 2), smooth=0.0001):
    """
    Sorenson (Soft) Dice loss
    Using -log(Dice) as the loss since it is better behaved.
    Also, the log allows avoidance of the division which
    can help prevent underflow when the numbers are very small.
    """
    # NOT OHE, for 2D U-net
    # intersection = tf.reduce_sum(prediction * target, axis=axis)
    # p = tf.reduce_sum(prediction, axis=axis)
    # t = tf.reduce_sum(target, axis=axis)
    # numerator = tf.reduce_mean(intersection + smooth)
    # denominator = tf.reduce_mean(t + p + smooth)
    # dice_loss = -tf.math.log(2.*numerator) + tf.math.log(denominator)


    target = K.reshape(target,shape=(-1,param.num_classes))
    prediction = K.reshape(prediction,shape=(-1,param.num_classes))

    target= target[:,1:]
    prediction= prediction[:,1:]

    # sum_p=K.sum(prediction,axis=0)
    # sum_r=K.sum(target,axis=0)
    # sum_pr=K.sum(target * prediction,axis=0)
    # dice_numerator =2*sum_pr
    # dice_denominator =sum_r+sum_p
    # dice_score =(dice_numerator+K.epsilon() )/(dice_denominator+K.epsilon())
    # dice_loss = 1 - dice_score

    intersection=K.sum(target * prediction,axis=0)  
    t=K.sum(target,axis=0)
    p=K.sum(prediction,axis=0)
    numerator = K.sum(intersection + smooth)
    denominator = K.sum(p + t + smooth)
    dice_loss = -tf.math.log(2.*numerator) + tf.math.log(denominator)

    # intersection = K.sum(prediction * target, axis=-2)
    # p = K.sum(prediction, axis=-2)
    # t = K.sum(target, axis=-2)
    # numerator = K.mean(intersection + smooth)
    # denominator = K.mean(t + p + smooth)
    # dice_loss = -tf.math.log(2.*numerator) + tf.math.log(denominator)

    return dice_loss

def combined_dice_ce_loss(target, prediction, axis=(1, 2), smooth=0.0001):
    """
    Combined Dice and Binary Cross Entropy Loss
    """
    return (0.85)*dice_coef_loss(target, prediction, axis, smooth) + \
        (1-0.85)*tf.keras.losses.categorical_crossentropy(target, prediction)



#######################################################################################################################################



def weighted_log_loss(y_true, y_pred):
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # weights are assigned in this order : normal,necrotic,edema,enhancing 
    weights=np.array([1,5,2,4])
    # weights = K.variable(weights)
    loss = y_true * K.log(y_pred) * weights
    loss = K.mean(-K.sum(loss, -1))
    return loss

def gen_dice_loss(y_true, y_pred):
    '''
    computes the sum of two losses : generalised dice loss and weighted cross entropy
    '''
    #generalised dice score is calculated as in this paper : https://arxiv.org/pdf/1707.03237
    y_true_f = K.reshape(y_true,shape=(-1,param.num_classes))
    y_pred_f = K.reshape(y_pred,shape=(-1,param.num_classes))
    sum_p=K.sum(y_pred_f,axis=-2)
    sum_r=K.sum(y_true_f,axis=-2)
    sum_pr=K.sum(y_true_f * y_pred_f,axis=-2)
    weights=K.pow(K.square(sum_r)+K.epsilon(),-1)
    generalised_dice_numerator =2*K.sum(weights*sum_pr)
    generalised_dice_denominator =K.sum(weights*(sum_r+sum_p))
    generalised_dice_score =generalised_dice_numerator /generalised_dice_denominator
    GDL=1-generalised_dice_score
    del sum_p,sum_r,sum_pr,weights

    return GDL+weighted_log_loss(y_true,y_pred)






