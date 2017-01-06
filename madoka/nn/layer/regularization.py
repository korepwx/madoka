# -*- coding: utf-8 -*-
from tensorflow.contrib import slim

from madoka.utils.tfhelper import Bookkeeper

__all__ = ['dropout']


def dropout(inputs, keep_prob, noise_shape=None, name='Dropout'):
    """Create a dropout op applied to the inputs.

    With probability `keep_prob`, outputs the input element scaled up by
    `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the
    expected sum is unchanged.

    Parameters
    ----------
    inputs : tf.Tensor
        The input tensor.

    keep_prob : float | tf.Tensor
        The probability of each element to be kept.

    noise_shape : tuple[int]
        The shape for randomly generated keep/drop flags.

    name : str
        Alternative name for the dropout op.
    """
    is_training = Bookkeeper.for_graph().is_training
    return slim.dropout(inputs, keep_prob=keep_prob, noise_shape=noise_shape,
                        is_training=is_training, scope=name)
