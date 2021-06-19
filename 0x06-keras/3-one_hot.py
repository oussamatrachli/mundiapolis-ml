
#!/usr/bin/env python3
import tensorflow.keras as keras


def one_hot(labels, classes=None):
    oh_encode = keras.utils.to_categorical(labels, num_classes=classes)
    return oh_encode
