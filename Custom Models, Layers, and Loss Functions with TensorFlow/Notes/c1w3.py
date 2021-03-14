import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Lambda, Layer
from tensorflow.keras.models import Model

# lambda layer =================================================================
# eg: use lambda abs to replace relu

tf.keras.layers.Lambda(lambda x: tf.abs(x))

from keras import backend as K
def my_relu(x):
    return K.maximum(0.5, x)
# tf.keras.layers.Lambda(my_relu)

# custom layer (trainable) =====================================================
class SimpleDense(Layer):
    def __init__(self, units=32):
        super(SimpleDense, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(name='kernel', initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float32'), trainable=True)
        
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(name='bias', initial_value=b_init(shape=(self.units,), dtype='float32'), trainable=True)
    
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

my_dense = SimpleDense(units=1)
x = tf.ones((1,1))
y = my_dense(x)
print(my_dense.variables)

# custom layer with activation =================================================
# can lambda layer for activation or modify layer class
class SimpleDense(Layer):
    def __init__(self, units=32, activation=None):
        super(SimpleDense, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)
    
    def build(self, input_shape):
        # same from above
        pass
    
    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.w) + self.b)
