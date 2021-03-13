import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Lambda
from tensorflow.keras.models import Model

# Custom loss function =========================================================
# Huber Loss, quadratic if error is small, linear if error is big

def my_huber_loss(y_true, y_pred):
    threshold = 1
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= threshold
    small_error_loss = tf.square(error) / 2
    big_error_loss = threshold * (tf.abs(error) - (threshold * 0.5))
    return tf.where(is_small_error, small_error_loss, big_error_loss)

# model.compile(optimizer='sgd', loss=my_huber_loss)

def my_huber_loss_with_threshold(threshold):
    def my_huber_loss(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= threshold
        small_error_loss = tf.square(error) / 2
        big_error_loss = threshold * (tf.abs(error) - (threshold * 0.5))        
        return tf.where(is_small_error, small_error_loss, big_error_loss)
    return my_huber_loss

# model.compile(optimizer='sgd', loss=my_huber_loss_with_threshold(threshold=1))

# Loss class setup =============================================================

from tensorflow.keras.losses import Loss

class MyHuberLoss(Loss):
    threshold = 1
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
    
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= self.threshold
        small_error_loss = tf.square(error) / 2
        big_error_loss = self.threshold * (tf.abs(error) - (self.threshold * 0.5))    
        return tf.where(is_small_error, small_error_loss, big_error_loss)

# model.compile(optimizer='sgd', loss=MyHuberLoss(threshold=1))

# Contrastive Loss =============================================================
class ContrastiveLoss(Loss):
    margin = 0
    def __init__(self, margin):
        super().__init__()
        self.margin = margin
    
    def call(self, y_true, y_pred):
        square_pred = keras.square(y_pred)
        margin_square = keras.square(keras.maximum(self.margin - y_pred, 0))
        return keras.mean(y_true * square_pred + (1 - y_true) * margin_square)