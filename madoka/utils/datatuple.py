# -*- coding: utf-8 -*-
import numpy as np

from . import misc

__all__ = ['DataTuple']


class DataTuple(object):
    """A tuple of numpy arrays, which indicates features and labels.

    In supervised learning, data usually consist of a feature array,
    paired with a label array.  However, in unsupervised learning,
    there is usually only one feature array.  Sometimes there might
    be extraordinary needs to make use of more than two arrays.

    This class then provides a general abstraction for associated
    data arrays, making ``DataFlow`` and ``DataWindow``
    free from the difference.

    Parameters
    ----------
    *arrays : collections.Iterable[numpy.ndarray]
        A list of numpy arrays.
        Lengths of the first dimension of all given arrays must be same.
    """

    def __init__(self, *arrays):
        for a in arrays:
            if not isinstance(a, np.ndarray):
                raise TypeError('%r is not a numpy array.' % (a,))
        if len(misc.unique([len(a) for a in arrays])) > 1:
            raise RuntimeError('Lengths of first dimension do not match.')
        self._arrays = list(arrays)

    def as_tuple(self):
        """Get arrays as tuple."""
        return tuple(self._arrays)

    @property
    def X(self):
        """Get the 1st array."""
        return self._arrays[0]

    @property
    def y(self):
        """Get the 2nd array."""
        return self._arrays[1]

    @property
    def data_count(self):
        """Get the count of data."""
        return len(self._arrays[0])

    def __len__(self):
        return len(self._arrays)

    def __iter__(self):
        return iter(self._arrays)

    def __getitem__(self, item):
        return self._arrays[item]

    def __setitem__(self, key, value):
        self._arrays[key] = value

    def __repr__(self):
        return 'DataTuple(%s)' % ','.join('%r' % (a,) for a in self._arrays)
