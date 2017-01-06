# -*- coding: utf-8 -*-
import six
import tensorflow as tf

__all__ = ['make_placeholder_for']


def make_placeholder_for(name, data, dtype=None, collection_keys=None):
    """Make a placeholder for specified data array.

    The constructed placeholder will have the shape (None,) + data.shape[1:],
    and the same dtype as the data, unless a different one is given.

    Parameters
    ----------
    name : str
        Name of the placeholder.

    data : numpy.ndarray
        Data array that should later feed into this placeholder.

    dtype : str | numpy.dtype | tf.DType
        Specify a data type other than what the data gives.
  
    collection_keys : str | list[str]
        Collections where this placeholder should be in.

    Returns
    -------
    tf.Tensor
        The placeholder tensor.
    """
    if dtype is None:
        dtype = tf.as_dtype(data.dtype)
    elif not isinstance(dtype, tf.DType):
        dtype = tf.as_dtype(dtype)
    ph = tf.placeholder(
        name=name,
        shape=(None,) + data.shape[1:],
        dtype=dtype
    )
    if collection_keys:
        if isinstance(collection_keys, six.string_types):
            collection_keys = [collection_keys]
        for key in collection_keys:
            tf.add_to_collection(key, ph)
    return ph
