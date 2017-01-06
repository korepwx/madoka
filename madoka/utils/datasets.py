# -*- coding: utf-8 -*-
"""Utilities to load well-known datasets."""
import gzip
import os

import numpy as np
import six

from .datatuple import DataTuple

if six.PY2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve

__all__ = [
    'get_cache_dir',
    'cached_download',
    'load_mnist',
]


def get_cache_dir(name, root_dir=None):
    """Get the cache directory for downloading a particular data set.

    Parameters
    ----------
    name : str
        Name of the data set.

    root_dir : str
        Root directory for cache.
        If not specified, will automatically choose one according to OS.

    Returns
    -------
    str
        Path of the data set cache directory.
    """
    if root_dir is None:
        root_dir = os.path.expanduser('~/.madoka/dataset')
    return os.path.join(root_dir, name)


def cached_download(uri, cache_file):
    """Download ``uri`` with caching.

    Parameters
    ----------
    uri : str
        URI to be downloaded.

    cache_file : str
        Path of the cache file.

    Returns
    -------
    str
        The full path of the downloaded file.
    """
    cache_file = os.path.abspath(cache_file)
    if not os.path.isfile(cache_file):
        parent_dir = os.path.split(cache_file)[0]
        if not os.path.isdir(parent_dir):
            os.makedirs(parent_dir)

        tmp_file = '%s~' % cache_file
        try:
            urlretrieve(uri, tmp_file)
            os.rename(tmp_file, cache_file)
        finally:
            if os.path.isfile(tmp_file):
                os.remove(tmp_file)
    return cache_file


def load_mnist(cache_dir=None, flatten_images=False, as_float=True, dtype=None,
               label_dtype=None):
    """Download mnist training and testing data as numpy array.

    Parameters
    ----------
    cache_dir : str
        Path to the cache directory.  Will automatically choose one according
        to ``get_cache_dir`` if not specified.

    flatten_images : bool
        If True, flatten images to 1D vectors.
        If False, shape the images to 3D tensors of shape (28, 28, 1),
        where the last dimension is the greyscale channel.

    as_float : bool
        If True, scale the byte pixels to 0.0~1.0 float numbers.
        If False, keep the byte pixels in 0~255 byte numbers.

    dtype : str | numpy.dtype
        Cast the image pixels into this type.

        If not specified, will use ``madoka.config.floatX`` if ``as_float``
        is True, or ``numpy.uint8`` if ``as_float`` is False.

    label_dtype : str | numpy.dtype
        Cast the image labels into this type.
        If not specified, will use ``numpy.uint8``.

    Returns
    -------
    (DataTuple, DataTuple)
        Training set and testing set.
    """
    from madoka import config
    cache_dir = cache_dir or get_cache_dir('mnist')
    root_uri = 'http://yann.lecun.com/exdb/mnist/'

    def load_mnist_images(filename):
        cache_file = os.path.join(cache_dir, filename)
        cache_file = cached_download(root_uri + filename, cache_file)
        with gzip.open(cache_file, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)

        if flatten_images:
            data = data.reshape(-1, 784)
        else:
            data = data.reshape(-1, 28, 28, 1)

        if as_float:
            data = data / np.array(256, dtype=dtype or config.floatX)
        elif dtype is not None:
            data = np.asarray(data, dtype=dtype)
        else:
            data = np.asarray(data, dtype=np.uint8)

        return data

    def load_mnist_labels(filename):
        cache_file = os.path.join(cache_dir, filename)
        cache_file = cached_download(root_uri + filename, cache_file)
        with gzip.open(cache_file, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        if label_dtype is not None:
            data = data.astype(label_dtype)
        return data

    # We can now download and read the training and test set images and labels.
    train_X = load_mnist_images('train-images-idx3-ubyte.gz')
    train_y = load_mnist_labels('train-labels-idx1-ubyte.gz')
    test_X = load_mnist_images('t10k-images-idx3-ubyte.gz')
    test_y = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    return DataTuple(train_X, train_y), DataTuple(test_X, test_y)
