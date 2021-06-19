#!/usr/bin/env python3

import tensorflow.keras as keras


def save_config(network, filename):
    json_network = network.to_json()
    with open(filename, 'w') as f:
        f.write(json_network)
    return None


def load_config(filename):
    with open(filename, 'r') as f:
        json_saved = f.read()
    return keras.models.model_from_json(json_saved)
