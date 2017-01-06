# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from madoka.utils import tfhelper
from .base import Layer
from .constraints import SingleOutputLayer
from ..init import Initializer, XavierNormal, Constant
from ..nonlinearity import NonLinearity, Rectify

__all__ = ['DenseLayer']


class DenseLayer(Layer, SingleOutputLayer):
    """Fully connected layer.

    Parameters
    ----------
    incoming : tf.Tensor | tuple[tf.Tensor] | Layer | tuple[Layer]
        Tensor(s) or layer(s) that feed into this layer.

    num_units : int
        Number of hidden units in this softmax layer.

    W : Initializer | numpy.ndarray | tf.Variable
        Initializer, numpy array or another variable, as the weight parameter.

    b : Initializer | numpy.ndarray | tf.Variable
        Initializer, numpy array or another variable, as the bias parameter.
        If set to None, then no bias will be applied.

    nonlinearity : NonLinearity | (tf.Tensor) -> tf.Tensor
        Non-linear function or function object.
        If set to None, no non-linear function will be applied.

    name : str
        Name of this layer.

    scope : tf.VariableScope
        Scope for the variables created in this layer.

    Attributes
    ----------
    num_units : int
        Number of hidden units in this layer.

    nonlinearity : (tf.Tensor) -> tf.Tensor
        Non-linear function that should be applied to the layer activation.

    W : tf.Variable
        The weights.

    b : tf.Variable
        The bias.

    activation : tf.Tensor
        Linear activation of this layer, before non-linearity is applied.
    """

    def __init__(self, incoming, num_units, W=XavierNormal(), b=Constant(0.),
                 nonlinearity=Rectify(), name=None):
        super(DenseLayer, self).__init__(incoming=incoming, name=name)
        self.num_units = num_units
        self.nonlinearity = nonlinearity

        num_inputs = int(np.prod(self.input_shape[1:]))
        self.W = self.add_param(W,
                                name='W',
                                shape=(num_inputs, num_units),
                                trainable=True)
        if b is None:
            self.b = b
        else:
            self.b = self.add_param(b,
                                    name='b',
                                    shape=(num_units,),
                                    trainable=True)

        # compose the linear activation
        if isinstance(incoming, Layer):
            incoming = incoming.output
        with self.variable_space('activation'):
            # flatten or expand the incoming tensor into 2-d tensor.
            if incoming.get_shape().ndims > 2:
                incoming = tfhelper.flatten(incoming, ndim=2)
            elif incoming.get_shape().ndims == 1:
                incoming = tf.reshape(incoming, [-1, 1])
            # compute the linear activation
            self.activation = tf.matmul(incoming, self.W)
            if self.b is not None:
                self.activation += self.b

        # compose the non-linear output
        if self.nonlinearity is not None:
            with self.variable_space('output'):
                self._output = self.nonlinearity(self.activation)
        else:
            self._output = self.activation

    @property
    def output(self):
        return self._output
