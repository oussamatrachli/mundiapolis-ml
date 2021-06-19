#!/usr/bin/env python3

import tensorflow.keras as keras


def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, verbose=True, shuffle=False):
    stopping = []
    if early_stopping and validation_data:
        stopping.append(keras.callbacks.EarlyStopping(patience=patience))
    history = network.fit(data, labels, epochs=epochs, batch_size=batch_size, shuffle=shuffle, verbose=verbose, callbacks=stopping, validation_data=validation_data)
    return history
