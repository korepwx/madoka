# -*- coding: utf-8 -*-
from itertools import chain

from madoka.utils import minibatch_indices_iterator, DEFAULT_PREDICT_BATCH_SIZE
from .base import DataFlow

__all__ = []


class _ArrayLike(object):
    """Array-like object to access data flow.

    This class acts as a proxy to access one internal array of a data flow.

    Parameters
    ----------
    flow : DataFlow
        Data flow instance.

    array_idx : int
        The index of the array in the data flow.
    """

    def __init__(self, flow, array_idx):
        self._flow = flow
        self._array_idx = array_idx
        self._array_indices = (array_idx,)
        self._dtype = self._flow.data_types[self._array_idx]
        self._shape = ((self._flow.epoch_size,) +
                       self._flow.data_shapes[self._array_idx])

    def __getitem__(self, item):
        return self._flow.get(item, array_indices=self._array_indices)[0]

    def __len__(self):
        return self._flow.epoch_size

    def __iter__(self):
        # We fetch data from the underlying data flow by mini-batches.
        # This may help to reduce the overhead of calling `DataFlow.get`,
        # without having to prefetch all of the data into memory at once.
        return chain(*(
            self._flow.get(batch, self._array_indices)[0]
            for batch in minibatch_indices_iterator(
                self._flow.epoch_size,
                DEFAULT_PREDICT_BATCH_SIZE
            )
        ))

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    def to_numpy(self):
        """Fetch all data and construct a numpy array."""
        return self._flow.get(slice(0, None, None), self._array_indices)[0]
