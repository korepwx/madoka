# -*- coding: utf-8 -*-
import numpy as np
import six
import tensorflow as tf

__all__ = [
    'flatten', 'l1_reg', 'l2_reg', 'apply_tensor_weight',
    'make_ensemble_output', 'make_ensemble_classifier',
]


def flatten(x, ndim=1, name='Flatten'):
    """Flatten x into ndim dimensions.

    Returns a view of x with ndim dimensions, whose shape for the first ndim-1
    dimensions will be the same as x, and shape in the remaining dimension will
    be expanded to fit in all the data from x.
    """
    with tf.name_scope(name, values=[x]):
        shape = x.get_shape()
        total_dim = len(shape)

        if total_dim == ndim:
            return x
        elif total_dim < ndim:
            raise ValueError('Attempt to flatten "x" to %r dimensions, but "x" '
                             'only has %r dimensions.' % (ndim, total_dim))

        if shape.is_fully_defined():
            # all the dimensions are fixed, thus we can use the static shape.
            shape = shape.as_list()[:ndim - 1] + [-1]
        else:
            # the shape is dynamic, so we have to generate a dynamic flatten
            # shape.
            shape = tf.concat(0, [tf.shape(x)[:ndim - 1], [-1]])

        return tf.reshape(x, shape)


def l1_reg(tensors, name='L1Reg'):
    """Compute the L1 regularization term for given tensors.

        l1(x) = sum_i | x_i |

    Parameters
    ----------
    tensors : tf.Variable | tf.Tensor | tuple[tf.Variable] | tuple[tf.Tensor]
        Tensors whose L1 regularization term should be computed.

    name : str
        Name of this operation in the graph.

    Returns
    -------
    tf.Tensor
    """
    if hasattr(tensors, '__iter__'):
        tensors = list(tensors)
        with tf.name_scope(name, values=tensors):
            return sum(tf.reduce_sum(tf.abs(p)) for p in tensors)
    else:
        with tf.name_scope(name, values=[tensors]):
            return tf.reduce_sum(tf.abs(tensors))


def l2_reg(tensors, name='L2Reg'):
    """Compute the L2 regularization term for given tensors.

        l2(x) = \frac{1}{2} sum_i x_i**2

    Parameters
    ----------
    tensors : tf.Variable | tf.Tensor | tuple[tf.Variable] | tuple[tf.Tensor]
        Tensors whose L2 regularization term should be computed.

    name : str
        Name of this operation in the graph.

    Returns
    -------
    tf.Tensor
    """
    if hasattr(tensors, '__iter__'):
        tensors = list(tensors)
        with tf.name_scope(name, values=tensors):
            return 0.5 * sum(tf.reduce_sum(p ** 2) for p in tensors)
    else:
        with tf.name_scope(name, values=[tensors]):
            return tf.nn.l2_loss(tensors)


def apply_tensor_weight(tensor, weight, name='ApplyTensorWeight'):
    """Apply element-wise weight to tensors.

    Parameters
    ----------
    tensor : tf.Tensor
        The tensor, whose dimension >= 1.

    weight : list | numpy.ndarray | tf.Tensor
        The element-wise weight as 1D tensor.

    name : str
        Name of this operation.

    Returns
    -------
    tf.Tensor
    """
    with tf.name_scope(name, values=[tensor, weight]):
        if not isinstance(weight, tf.Tensor):
            weight = tf.convert_to_tensor(weight, dtype=tensor.dtype)
        elif weight.dtype != tensor.dtype:
            weight = tf.cast(weight, tensor.dtype)

        # adjust the weight for multiplication
        weight_shape = weight.get_shape()
        tensor_shape = tensor.get_shape()

        if weight_shape.ndims != 1:
            raise ValueError('`weight` tensor must be 1-dimensional.')
        if tensor_shape.ndims < 1:
            raise ValueError('Expected tensor, but got scalar %r.' % tensor)

        if weight_shape.ndims != tensor_shape.ndims:
            shape_append = [1] * (tensor_shape.ndims - weight_shape.ndims)
            if weight_shape.is_fully_defined():
                weight_shape = weight_shape.as_list() + shape_append
            else:
                weight_shape = tf.concat(0, [weight_shape, shape_append])
            weight = tf.reshape(weight, weight_shape)

        # now multiply the loss with the weight
        return tf.mul(tensor, weight)


def make_ensemble_output(outputs, weight=None, name='MakeEnsembleOutput'):
    """Make ensemble output.

    Parameters
    ----------
    outputs : tf.Tensor | collections.Iterable[tf.Tensor]
        The ensemble components.

    weight : tf.Tensor | collections.Iterable[float | tf.Tensor]
        The weight of each component.
        If not specified, will set the weight of each component to one.

    name : str
        Name of this operation.

    Returns
    -------
    tf.Tensor
    """
    with tf.name_scope(name):
        if weight is not None:
            try:
                total_weight = sum(weight)
                if not isinstance(total_weight, (float,) + six.integer_types):
                    raise TypeError()
                if total_weight == 0:
                    raise ValueError(
                        'Weights sum up to 0, which is not allowed.')
                if isinstance(total_weight, six.integer_types):
                    total_weight = float(total_weight)
                weight = np.asarray(weight)
            except TypeError:
                if not isinstance(weight, tf.Tensor):
                    weight = tf.pack(list(weight))
                total_weight = tf.reduce_sum(weight)

            if not isinstance(outputs, tf.Tensor):
                outputs = tf.pack(list(outputs))

            # apply the weights to the output.
            outputs = apply_tensor_weight(
                outputs, weight, name='ApplyEnsembleWeight')
            outputs = tf.reduce_sum(outputs, [0]) / total_weight
        else:
            outputs = tf.reduce_mean(outputs, [0])
        return outputs


def make_ensemble_classifier(outputs,
                             weight=None,
                             name='MakeEnsembleClassifier'):
    """Make ensemble classifier.

    Parameters
    ----------
    outputs : tf.Tensor | collections.Iterable[tf.Tensor]
        The ensemble classification probability tensors.

    weight : tf.Tensor | collections.Iterable[float | tf.Tensor]
        The weight of each component classifier.
        If not specified, will set the weight of each component to one.

    name : str
        Name of this operation.

    Returns
    -------
    tf.Tensor
    """
    with tf.name_scope(name):
        outputs = tf.pack(list(outputs))
        output_shape = outputs.get_shape()
        if output_shape.ndims != 3:
            raise TypeError('The outputs should be 2-D probability '
                            'tensors.')
        depth = output_shape.as_list()[2]
        if depth is None:
            depth = tf.shape(outputs)[2]
        outputs = tf.one_hot(tf.argmax(outputs, 2), depth)
        return make_ensemble_output(outputs, weight)
