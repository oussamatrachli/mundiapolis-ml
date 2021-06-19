#!/usr/bin/env python3

import tensorflow.keras as keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    keras.Model()
    inputs = keras.Input(shape=(nx,))
    x = keras.layers.Dense(layers[0], activation=activations[0], kernel_regularizer=keras.regularizers.l2(lambtha))(inputs)
    y = x
    rate = 1 - keep_prob
    
    for i in range(1, len(layers)):
        if i == 1:
            y = keras.layers.Dropout(rate)(x)
        else:
            y = keras.layers.Dropout(rate)(y)
        y = keras.layers.Dense(layers[i], activation=activations[i], kernel_regularizer=keras.regularizers.l2(lambtha))(y)
    
    model = keras.Model(inputs=inputs, outputs=y)
    return model
