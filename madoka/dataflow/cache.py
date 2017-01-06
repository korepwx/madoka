# -*- coding: utf-8 -*-
import numpy as np

from .base import PipelineDataFlow

__all__ = []


class _EpochCacheDataFlow(PipelineDataFlow):
    """Data flow which caches of original flow every epoch."""

    def __init__(self, origin):
        super(_EpochCacheDataFlow, self).__init__(origin)
        self._cached_arrays = None  # type: np.ndarray
        self._build_cache()

    def _build_cache(self):
        self._cached_arrays = self._origin.all()
        for arr in self._cached_arrays:
            arr.setflags(write=False)

    def get(self, item, array_indices=None):
        if array_indices is not None:
            ret = tuple(self._cached_arrays[i][item] for i in array_indices)
        else:
            ret = tuple(arr[item] for arr in self._cached_arrays)
        return ret

    def all(self):
        return self._cached_arrays

    def get_array_like(self, array_idx):
        return self._cached_arrays[array_idx]

    def reset_epoch(self):
        super(_EpochCacheDataFlow, self).reset_epoch()
        self._build_cache()
        return self
