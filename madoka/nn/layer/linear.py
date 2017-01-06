# -*- coding: utf-8 -*-
import tensorflow as tf

from madoka.utils.tfhelper import apply_tensor_weight
from .base import Layer
from .constraints import SupervisedLossLayer
from .dense import DenseLayer
from ..init import Initializer, XavierNormal, Constant
from ..metric import SquareError

__all__ = ['LinearRegression']


class LinearRegression(DenseLayer, SupervisedLossLayer):
    """Linear regression layer with loss.

    Parameters
    ----------
    incoming : tf.Tensor | tuple[tf.Tensor] | Layer | tuple[Layer]
        Tensor(s) or layer(s) that feed into this layer.

    num_units : int
        Number of hidden units in this softmax layer.

        If set to 1, the layer output will be squeezed to 1-dimension,
        instead of having the shape of (?, 1).

    error_metric : (tf.Tensor, tf.Tensor) -> tf.Tensor
        Error metric for two tensors.

    error_scales : numpy.ndarray
        Scaling factor of each dimension in computed error.

        The shape of this scaling factor must match the shape of each element
        in the computed error, i.e., the expression `error_weights * error[i]`
        must be well-defined.

    W : Initializer | numpy.ndarray | tf.Variable
        Initializer, numpy array or another variable, as the weight parameter.

    b : Initializer | numpy.ndarray | tf.Variable
        Initializer, numpy array or another variable, as the bias parameter.

    name : str
        Name of this layer.

    Attributes
    ----------
    num_units, nonlinearity, W, b, activation
        Attributes inherited from ``Dense`` layer.

    error_metric : (tf.Tensor, tf.Tensor) -> tf.Tensor
        Error metric for two tensors.

    error_scales : numpy.ndarray
        Scaling factor of each dimension in computed error.

    Notes
    -----
    The aggregated loss of this layer should be computed by averaging all
    of the weighted error components.
    """

    def __init__(self, incoming, num_units, error_metric=SquareError(),
                 error_scales=None, W=XavierNormal(), b=Constant(0.),
                 name='LinearRegression'):
        super(LinearRegression, self).__init__(
            incoming=incoming, num_units=num_units, W=W, b=b, nonlinearity=None,
            name=name
        )
        self.error_metric = error_metric
        self.error_scales = error_scales

        # fix the output shape if num_units == 1, by squeeze the last dimension
        if num_units == 1:
            with self.variable_space('output'):
                self._output = tf.squeeze(self._output, squeeze_dims=[1])

        # prepend a dimension to the front of error_scales
        if error_scales is not None:
            self._error_scales_reshaped = \
                error_scales.reshape((1,) + error_scales.shape)
        else:
            self._error_scales_reshaped = None

    def get_loss(self, target_ph=None, weight=None, aggregate=False):
        self._validate_target(target_ph)

        # check the dimension of target placeholder and layer output
        if target_ph.get_shape().ndims != self.output.get_shape().ndims:
            raise TypeError('Dimension of `target_ph` does not match layer '
                            'output.')

        with self.variable_space('loss'):
            loss = self.error_metric(self.output, target_ph)
            if self._error_scales_reshaped:
                # TODO: can we simply do type casting to loss.dtype here?
                # For example, what if loss.dtype == int?  Is it possible?
                # loss = loss * tf.cast(self._error_scales_reshaped, loss.dtype)
                loss = loss * self._error_scales_reshaped
            if weight is not None:
                loss = apply_tensor_weight(loss, weight, name='ApplyLossWeight')
            if aggregate:
                loss = tf.reduce_mean(loss)
            return loss
