# -*- coding: utf-8 -*-
import numpy as np

from .base import DataFlow

__all__ = []


class _NumpyDataFlow(DataFlow):
    """Data flow backed by in-memory Numpy arrays.

    Parameters
    ----------
    arrays : collection.Iterable[np.ndarray]
        The numpy arrays.
    """

    @staticmethod
    def _make_array_immutable(arr):
        arr = arr[:]
        arr.setflags(write=False)
        return arr

    def __init__(self, arrays):
        super(_NumpyDataFlow, self).__init__()

        if isinstance(arrays, np.ndarray):
            arrays = (self._make_array_immutable(arrays),)
        else:
            arrays = tuple(self._make_array_immutable(arr) for arr in arrays)
        if not arrays:
            raise TypeError('No array.')
        if len(set(len(arr) for arr in arrays)) > 1:
            raise TypeError('Array lengths mismatch.')
        if any(not isinstance(arr, np.ndarray) for arr in arrays):
            raise TypeError('Not all arrays are numpy array.')
        self._arrays = arrays

    @property
    def numpy_arrays(self):
        """Get the underlying numpy arrays.

        Returns
        -------
        tuple[np.ndarray]
        """
        return self._arrays

    @property
    def input_flows(self):
        return ()

    @property
    def array_count(self):
        return len(self._arrays)

    @property
    def epoch_size(self):
        return len(self._arrays[0])

    @property
    def data_shapes(self):
        return tuple(arr.shape[1:] for arr in self._arrays)

    @property
    def data_types(self):
        return tuple(arr.dtype for arr in self._arrays)

    @property
    def is_constant(self):
        return True

    def reset_epoch(self):
        return self

    def get(self, item, array_indices=None):
        if array_indices is not None:
            ret = tuple(self._arrays[i][item] for i in array_indices)
        else:
            ret = tuple(arr[item] for arr in self._arrays)
        return ret

    def all(self):
        return self._arrays

    def get_array_like(self, array_idx):
        return self._arrays[array_idx]

    def sliced(self, slice_):
        return _NumpyDataFlow(a[slice_] for a in self._arrays)

    def indexed(self, indices):
        return _NumpyDataFlow(a[indices] for a in self._arrays)

    def masked(self, masks):
        return _NumpyDataFlow(a[masks] for a in self._arrays)

    def select_array(self, array_indices):
        if hasattr(array_indices, '__iter__'):
            array_indices = tuple(array_indices)
        else:
            array_indices = (array_indices,)
        if not array_indices:
            raise TypeError('No arrays are selected.')
        return _NumpyDataFlow(self._arrays[i] for i in array_indices)
