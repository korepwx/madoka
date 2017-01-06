# -*- coding: utf-8 -*-
"""Wrappers on TensorFlow nonlinearity functions, to ease serialization."""

import tensorflow as tf

__all__ = [
    'Sigmoid', 'Softmax', 'Tanh', 'Rectify'
]


class NonLinearity(object):
    """Base class for activation function wrappers."""


class Sigmoid(NonLinearity):
    def __call__(self, activation):
        return tf.nn.sigmoid(activation)


class Softmax(NonLinearity):
    def __call__(self, activation):
        return tf.nn.softmax(activation)


class Tanh(NonLinearity):
    def __call__(self, activation):
        return tf.nn.tanh(activation)


class Rectify(NonLinearity):
    def __call__(self, activation):
        return tf.nn.relu(activation)
