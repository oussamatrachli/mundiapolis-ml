#!/usr/bin/env python3

import tensorflow.keras as keras


def train_model(network, data, labels, batch_size, epochs, validation_data=None, verbose=True, shuffle=False):
    return network.fit(data, labels, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle, validation_data=validation_data)
