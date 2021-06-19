#!/usr/bin/env python3

import tensorflow.keras as keras

def save_model(network, filename):
    keras.models.save_model(model=network, filepath=filename)
    return None

def load_model(filename):
    return keras.models.load_model(filepath=filename)
