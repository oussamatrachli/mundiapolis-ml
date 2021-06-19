#!/usr/bin/env python3

import tensorflow.keras as keras


def optimize_model(network, alpha, beta1, beta2):
    opt = keras.optimizers.Adam(alpha, beta_1=beta1, beta_2=beta2)
    network.compile(loss='categorical_crossentropy', optimizer=opt,
                    metrics=['accuracy'])
    return None
