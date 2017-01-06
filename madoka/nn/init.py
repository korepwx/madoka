# -*- coding: utf-8 -*-
"""Wrappers on TensorFlow initializers, to ease object serialization."""

import math

import six
import tensorflow as tf

from madoka import config

__all__ = [
    'Initializer', 'Constant', 'Uniform', 'Normal', 'TruncatedNormal',
    'XavierNormal', 'XavierUniform',
]


class Initializer(object):
    """Base class for all TensorFlow initializer wrappers."""

    def __init__(self, seed=None, dtype=None):
        if dtype is None:
            dtype = tf.as_dtype(config.floatX)
        elif not isinstance(dtype, tf.DType):
            dtype = tf.as_dtype(dtype)
        self.seed = seed
        self.dtype = dtype

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        return self._tf_init(shape, dtype, partition_info)

    @property
    def _tf_init(self):
        """TensorFlow initializer instance."""
        raise NotImplementedError()


class Constant(Initializer):
    """Constant initializer wrapper."""

    def __init__(self, value=0, dtype=None):
        super(Constant, self).__init__(dtype=dtype)
        self.value = value

    @property
    def _tf_init(self):
        return tf.constant_initializer(value=self.value, dtype=self.dtype)


class Uniform(Initializer):
    """Uniform initializer wrapper."""

    def __init__(self, minval=0, maxval=None, seed=None, dtype=None):
        super(Uniform, self).__init__(seed=seed, dtype=dtype)
        self.minval = minval
        self.maxval = maxval

    @property
    def _tf_init(self):
        return tf.random_uniform_initializer(minval=self.minval,
                                             maxval=self.maxval,
                                             seed=self.seed,
                                             dtype=self.dtype)


class Normal(Initializer):
    """Normal initializer wrapper."""

    def __init__(self, mean=0.0, stddev=1.0, seed=None, dtype=None):
        super(Normal, self).__init__(seed=seed, dtype=dtype)
        self.mean = mean
        self.stddev = stddev

    @property
    def _tf_init(self):
        return tf.random_normal_initializer(mean=self.mean, stddev=self.stddev,
                                            seed=self.seed, dtype=self.dtype)


class TruncatedNormal(Initializer):
    """Truncated normal initializer wrapper.

    Values sampled from this initializer is similar to those from ``Normal``
    initializer, except that any value beyond two standard deviations are
    discarded and re-drawn.
    """

    def __init__(self, mean=0.0, stddev=1.0, seed=None, dtype=None):
        super(TruncatedNormal, self).__init__(seed=seed, dtype=dtype)
        self.mean = mean
        self.stddev = stddev

    @property
    def _tf_init(self):
        return tf.truncated_normal_initializer(mean=self.mean,
                                               stddev=self.stddev,
                                               seed=self.seed,
                                               dtype=self.dtype)


class _XavierBase(Initializer):
    """Base xavier initializer.

    Parameters
    ----------
    gain : float | str
        Scaling factor for the weights.  Set this to 1.0 for linear and sigmoid
        units, to 'relu' or sqrt(2) for rectified linear units.  Other transfer
        functions may need different factors.
    """

    def __init__(self, gain=1.0, seed=None, dtype=None):
        if isinstance(gain, six.string_types):
            if gain == 'relu':
                gain = math.sqrt(2)
            else:
                raise ValueError('Unrecognized gain %r.' % gain)
        self.gain = gain
        super(_XavierBase, self).__init__(seed=seed, dtype=dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        if len(shape) < 2:
            raise RuntimeError('Xavier initializer only works with shapes '
                               'of length >= 2: got %r.' % shape)
        if len(shape) > 2:
            raise NotImplementedError()
        # This method is based on Lasagne implementation.
        n_inputs, n_outputs = shape[0], shape[-1]
        receptive_field_size = 1

        return self._sample(n_inputs, n_outputs, receptive_field_size,
                            shape, dtype, partition_info)

    def _sample(self, n_inputs, n_outputs, receptive_field_size, shape, dtype,
                partition_info):
        raise NotImplementedError()


class XavierNormal(_XavierBase):
    """Xavier weight initializer with normal distribution."""

    def _sample(self, n_inputs, n_outputs, receptive_field_size, shape, dtype,
                partition_info):
        with tf.name_scope('xavier_normal_initializer'):
            stddev = self.gain * math.sqrt(
                3.0 / ((n_inputs + n_outputs) * receptive_field_size))
            return tf.truncated_normal(shape=shape, mean=0.0, stddev=stddev,
                                       dtype=dtype, seed=self.seed)


class XavierUniform(_XavierBase):
    """Xavier weight initializer with uniform distribution."""

    def _sample(self, n_inputs, n_outputs, receptive_field_size, shape, dtype,
                partition_info):
        with tf.name_scope('xavier_uniform_initializer'):
            range = self.gain * math.sqrt(
                6.0 / ((n_inputs + n_outputs) * receptive_field_size))
            return tf.random_uniform(shape=shape, minval=-range, maxval=range,
                                     dtype=dtype, seed=self.seed)
