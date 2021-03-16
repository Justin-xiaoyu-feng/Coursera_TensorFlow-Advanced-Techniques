import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Lambda, Layer, Conv2D
from tensorflow.keras.models import Model
from keras import backend as K

# deep and wide model ==========================================================
# create model class
class WideAndDeepModel(Model):
    def __init__(self, units=30, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = Dense(units, activation=activation)
        self.hidden2 = Dense(units, activation=activation)
        self.main_output = Dense(1)
        self.aux_output = Dense(1)
    
    def call(self, inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(input_A)
        concat = concatenate([input_A, hidden2])
        main_output = self.main_output(concate)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output

# model = WideAndDeepModel()

# Built-in training, evaluation, and prediction loops ==========================
# model.fit()
# model.evaluate()
# model.predict()

# Saving and seralization APIs =================================================
# model.save()
# model.save_weights()

# Summarization and visualization APIs =========================================
# model.summary()
# tf.keras.utils.plot_model()


# ResNet =======================================================================
# CNN Residual Block
class CNNResidual(Layer):
    def __init__(self, layers, filters, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [Conv2D(filters, (3,3), activation='relu') for _ in range(layers)]
    
    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        return inputs + x

# DNN Residual Block
class DNNResidual(Layer):
    def __init__(self, layers, neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [Dense(neurons, activation='relu') for _ in range(layers)]
    
    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        return inputs + x
    
# Residual Network
class MyResidual(Model):
    def __init__(self, **kwargs):
        self.hidden1 = Dense(30, activation = 'relu')
        self.block1 = CNNResidual(2, 32)
        self.block2 = DNNResidual(2, 64)
        self.out = Dense(1)
    
    def call(self, inputs):
        x = self.hidden1(inputs)
        x = self.block1(x)
        for _ in range(3):
            x = self.block2(x)
        return self.out(x)