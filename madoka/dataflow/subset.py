# -*- coding: utf-8 -*-
import numpy as np

from madoka.utils import (slice_length, merge_slices, merge_slice_indices,
                          merge_slice_masks, is_mask_array)
from .base import DataFlow, PipelineDataFlow

__all__ = []


class _SlicedDataFlow(PipelineDataFlow):
    """Data flow sliced from another data flow.

    Parameters
    ----------
    origin : DataFlow
        Original data flow.

    slice_ : slice
        Slicing object.
    """

    def __init__(self, origin, slice_):
        super(_SlicedDataFlow, self).__init__(origin)
        if not isinstance(slice_, slice):
            raise TypeError('`slice_` is required to be a slice instance.')
        self._slice = slice_

    @property
    def epoch_size(self):
        return slice_length(self._o_epoch_size, self._slice)

    def get(self, item, array_indices=None):
        if isinstance(item, slice):
            item = merge_slices(self._o_epoch_size, self._slice, item)
        else:
            if not isinstance(item, np.ndarray):
                item = np.asarray(item)
            if is_mask_array(item):
                f = merge_slice_masks
            else:
                f = merge_slice_indices
            item = f(self._o_epoch_size, self._slice, item)
        return self._origin.get(item, array_indices=array_indices)

    def all(self):
        return self._origin.get(self._slice)

    def sliced(self, slice_):
        return self._origin.pipeline(_SlicedDataFlow,
                                     merge_slices(self._o_epoch_size,
                                                  self._slice, slice_))

    def indexed(self, indices):
        return self._origin.pipeline(_IndexedDataFlow,
                                     merge_slice_indices(self._o_epoch_size,
                                                         self._slice, indices))

    def masked(self, masks):
        return self._origin.pipeline(_IndexedDataFlow,
                                     merge_slice_masks(self._o_epoch_size,
                                                       self._slice, masks))


class _IndexedDataFlow(PipelineDataFlow):
    """Subset data flow according to indices.

    Parameters
    ----------
    origin : DataFlow
        Original data flow.

    indices : np.ndarray
        Indices of the subset, 1-d numpy array.
    """

    def __init__(self, origin, indices):
        super(_IndexedDataFlow, self).__init__(origin)
        if not isinstance(indices, np.ndarray):
            indices = np.asarray(indices)
        if is_mask_array(indices):
            raise TypeError('%r is not an indices array.' % indices)
        if len(indices.shape) != 1:
            raise TypeError('`indices` is required to be a 1-d numpy array.')
        self._indices = indices

    @property
    def epoch_size(self):
        return len(self._indices)

    def get(self, item, array_indices=None):
        return self._origin.get(self._indices[item],
                                array_indices=array_indices)

    def all(self):
        return self._origin.get(self._indices)

    def sliced(self, slice_):
        return self._origin.pipeline(_IndexedDataFlow, self._indices[slice_])

    def indexed(self, indices):
        return self._origin.pipeline(_IndexedDataFlow, self._indices[indices])

    def masked(self, masks):
        return self._origin.pipeline(_IndexedDataFlow, self._indices[masks])


class _MaskedDataFlow(_IndexedDataFlow):
    """Subset data flow according to masks.

    Parameters
    ----------
    origin : DataFlow
        Original data flow.

    masks : np.ndarray
        Masks of the subset, 1-d numpy array.
    """

    def __init__(self, origin, masks):
        if not isinstance(masks, np.ndarray):
            masks = np.asarray(masks)
        if not is_mask_array(masks):
            raise TypeError('%r is not a mask array.' % masks)
        if len(masks.shape) != 1:
            raise TypeError('`masks` is required to be a 1-d numpy array.')
        if len(masks) != origin.epoch_size:
            raise TypeError('Length of masks != data flow: %r != %r.' %
                            (len(masks), origin.epoch_size))
        super(_MaskedDataFlow, self).__init__(origin, np.where(masks)[0])


class _SelectArrayDataFlow(PipelineDataFlow):
    """Data flow with selected subset of arrays.

    Parameters
    ----------
    origin : DataFlow
        Original data flow.

    array_indices : int | collections.Iterable[int]
        Indices of the array subset.
    """

    def __init__(self, origin, array_indices):
        super(_SelectArrayDataFlow, self).__init__(origin)
        if hasattr(array_indices, '__iter__'):
            array_indices = tuple(array_indices)
        else:
            array_indices = (array_indices,)
        if not array_indices:
            raise TypeError('No arrays are selected.')
        if min(array_indices) < 0 or max(array_indices) >= self._o_array_count:
            raise IndexError('Array index out of range.')
        self._indices = array_indices
        self._unique_indices, self._indices_pos = \
            self._build_unique_indices_and_pos(self._indices)

    @staticmethod
    def _build_unique_indices_and_pos(indices):
        indices = tuple(indices)
        unique_indices = sorted(set(indices))
        mapping = {k: i for i, k in enumerate(unique_indices)}
        pos = tuple(mapping[k] for k in indices)
        return unique_indices, pos

    @property
    def array_count(self):
        return len(self._indices)

    @property
    def data_shapes(self):
        return tuple(self._o_data_shapes[i] for i in self._indices)

    @property
    def data_types(self):
        return tuple(self._o_data_types[i] for i in self._indices)

    def all(self):
        return self.get(slice(0, None, None))

    def get(self, item, array_indices=None):
        if array_indices is None:
            unique_indices, indices_pos = \
                self._unique_indices, self._indices_pos
        else:
            unique_indices, indices_pos = self._build_unique_indices_and_pos(
                self._indices[i] for i in array_indices)
        ret = self._origin.get(item, unique_indices)
        return tuple(ret[i] for i in indices_pos)
