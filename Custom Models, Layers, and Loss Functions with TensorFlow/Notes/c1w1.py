import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential

# Sequential API ===============================================================
seq_model = Sequential([
    layers.Flatten(input_shape = (28,28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Functional API ===============================================================
# Define the Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Lambda
from tensorflow.keras.models import Model

input = Input(shape=(28,28))
x = Flatten()(input)
x = Dense(128, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

func_model = Model(inputs=input, outputs=predictions)

# Multi-Output Model============================================================
input_layer = Input(shape=(8))
first_dense = Dense(128, activation='relu')(input_layer)
second_dense = Dense(128, activation='relu')(first_dense)

y1_output = Dense(1, name='y1_output')(second_dense)

third_dense = Dense(64, activation='relu')(second_dense)
y2_output = Dense(1, name='y2_output')(third_dense)

model = Model(inputs=input_layer, outputs=[y1_output, y2_output])
model.summary()

# Multi-Output Model============================================================
# Siamese network

def initialize_base_network():
    input_layer = Input(shape=(28,28,))
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(inputs=input, outputs=x)

base_network = initialize_base_network()
input_a = Input(shape=(28,28,))
input_b = Input(shape=(28,28,))

vect_output_a = base_network(input_a)
vect_output_b = base_network(input_b)

# output = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([vect_output_a, vect_output_b])

# model == Model([input_a, input_b], output)
# rms = RMSprop()
# model.complie(loss=contrastive_loss, optimizer = rms)