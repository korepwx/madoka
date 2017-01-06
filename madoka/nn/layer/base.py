# -*- coding: utf-8 -*-
from collections import OrderedDict

import numpy as np
import six
import tensorflow as tf

from madoka import config
from madoka.utils.tfhelper import ScopedObject
from ..init import Initializer

__all__ = ['Layer']


class Layer(ScopedObject):
    """Base class for non-deferred layers.

    A layer is a sub-structure of neural network.  In order to simplify the
    interfaces, all layers except ``DeferredLayer`` should operate on tensors
    directly, keeping the intermediate tensor results as instance members.
    This is different from the Lasagne approach, which would not actually
    compose the output tensors until `get_output_for` is called.

    Parameters
    ----------
    incoming : tf.Tensor | tuple[tf.Tensor] | Layer | tuple[Layer]
        Tensor(s) or layer(s) that feed into this layer.

    name : str
        Name of this layer.
    """

    def __init__(self, incoming, name=None):
        super(Layer, self).__init__(name=name)
        self.incoming = incoming
        self._params = OrderedDict()

    def get_params(self):
        """Get all the parameters defined directly in this layer.

        Returns
        -------
        tuple[tf.Variable]
        """
        return tuple(six.itervalues(self._params))

    def add_param(self, spec, name, shape, dtype=None, trainable=True,
                  *args, **kwargs):
        """Add a parameter to the layer.

        Parameters
        ----------
        spec : Initializer | numpy.ndarray | tf.Variable
            Initializer, numpy array or another variable, as the parameter.

            If it is an initializer or a numpy array, then a new variable
            will be created as the parameter.

        name : str
            Name of the new variable.

        shape : tuple[int] | list[int]
            Shape of the new variable.

        dtype : str | numpy.dtype | tf.DType
            Data type of the variable.

        trainable : bool
            Whether or not this variable is trainable?

        *args, **kwargs
            Additional arguments passed to ``tf.get_variable``.

        Returns
        -------
        tf.Variable
        """
        if name in self._params:
            raise KeyError('Duplicated variable name %r.' % name)
        if isinstance(spec, tf.Variable):
            self._params[name] = spec
        else:
            if dtype is None:
                dtype = tf.as_dtype(config.floatX)
            elif not isinstance(dtype, tf.DType):
                dtype = tf.as_dtype(dtype)
            if isinstance(spec, np.ndarray):
                if spec.shape != shape:
                    raise ValueError('Shape of initializer %r != %r.' %
                                     (spec, shape))
                # TensorFlow rejects explicit shape with constant initializer.
                shape = None
            with tf.variable_scope(self.scope):
                var = tf.get_variable(name=name,
                                      shape=shape,
                                      dtype=dtype,
                                      initializer=spec,
                                      trainable=trainable,
                                      *args,
                                      **kwargs)
                tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, var)
                self._params[name] = var
            return var

    @property
    def input_shape(self):
        """Get the input shape of this layer.

        Returns
        -------
        tuple[int] | tuple[tuple[int]]
            Shape of the incoming tensor(s).
        """
        def get_shape(input):
            if isinstance(input, Layer):
                input = input.output
            return input.get_shape()

        if isinstance(self.incoming, (tuple, list)):
            return tuple(get_shape(i) for i in self.incoming)
        return get_shape(self.incoming)

    @property
    def output(self):
        """Get the output tensor(s) of the layer.

        Returns
        -------
        tf.Tensor | tuple[tf.Tensor]
            The layer output(s).
        """
        raise NotImplementedError()

    def get_loss(self, target_ph=None, weight=None, aggregate=False):
        """Get the loss of the layer.

        Multiple calls to this method, even with the same ``target_ph``,
        is not guaranteed to return the same loss tensor.

        Parameters
        ----------
        target_ph : tf.Tensor
            Placeholder for the target labels.
            Required only if the layer is supervised layer.

        weight : tf.Tensor
            Element-wise loss weight as 1D tensor.

            If specified, the element-wise loss will be multiplied by this
            weight tensor.  You may use utilities from `madoka.nn`
            to compute a weight tensor based on the label placeholder.

        aggregate : bool
            If True, take the average of element-wise loss.
            If False, return the element-wise loss directly.

        Returns
        -------
        tf.Tensor
            The average loss scalar, or element-wise loss tensor.
        """
        raise RuntimeError('%s does not have a loss output.' %
                           self.__class__.__name__)
