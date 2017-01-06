# -*- coding: utf-8 -*-
import tensorflow as tf

from madoka import config

__all__ = ['loss_weight_by_class']


def loss_weight_by_class(labels, class_weight, dtype=config.floatX,
                         name='LossWeightByClass'):
    """Compute element-wise loss weight by class weight.

    Parameters
    ----------
    labels : tf.Tensor
        Class labels tensor of any shape.

    class_weight : list | numpy.ndarray | tf.Tensor
        Class weight as 1D tensor.

    dtype : str | tf.DType
        Data type of the returned loss weight tensor.

    name : str
        Name of this operation.

    Returns
    -------
    tf.Tensor
    """
    with tf.name_scope(name, values=[labels, class_weight]):
        if not isinstance(class_weight, tf.Tensor):
            class_weight = tf.convert_to_tensor(class_weight, dtype=dtype)
        elif class_weight.dtype != dtype:
            class_weight = tf.cast(class_weight, dtype)
        if class_weight.get_shape().ndims != 1:
            raise ValueError('`class_weight` is expected to be 1D tensor.')
        return tf.gather(class_weight, labels)
