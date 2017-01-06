# -*- coding: utf-8 -*-
import numpy as np

from madoka.utils import INT32_MAX_VALUE
from .subset import _IndexedDataFlow

__all__ = []


class _BaseShuffledDataFlow(_IndexedDataFlow):

    def __init__(self, origin):
        super(_BaseShuffledDataFlow, self).__init__(
            origin, self._make_random_indices(origin.epoch_size))

    @staticmethod
    def _make_random_indices(length):
        dtype = np.int32 if length <= INT32_MAX_VALUE else np.int64
        indices = np.arange(length, dtype=dtype)
        np.random.shuffle(indices)
        return indices


class _OneTimeShuffledDataFlow(_BaseShuffledDataFlow):
    """Data flow which is shuffled only once."""

    def __init__(self, origin):
        super(_OneTimeShuffledDataFlow, self).__init__(origin)


class _EpochShuffledDataFlow(_BaseShuffledDataFlow):
    """Data flow which is shuffled at every epoch."""

    def __init__(self, origin):
        super(_EpochShuffledDataFlow, self).__init__(origin)

    @property
    def is_constant(self):
        return False

    def reset_epoch(self):
        super(_EpochShuffledDataFlow, self).reset_epoch()
        self._indices = self._make_random_indices(self._o_epoch_size)
        return self


class _BaseResampledDataFlow(_IndexedDataFlow):

    def __init__(self, origin, sample_size=None):
        if sample_size is None:
            sample_size = origin.epoch_size
        super(_BaseResampledDataFlow, self).__init__(
            origin, self._make_random_indices(sample_size))
        self.sample_size = sample_size

    @staticmethod
    def _make_random_indices(length):
        dtype = np.int32 if length <= INT32_MAX_VALUE else np.int64
        indices = np.random.randint(0, length, size=length, dtype=dtype)
        return indices


class _OneTimeResampledDataFlow(_BaseResampledDataFlow):
    """Data flow which is resampled only once."""

    def __init__(self, origin, sample_size=None):
        super(_OneTimeResampledDataFlow, self).__init__(origin, sample_size)


class _EpochResampledDataFlow(_BaseResampledDataFlow):
    """Data flow which is resampled at every epoch."""

    def __init__(self, origin, sample_size=None):
        super(_EpochResampledDataFlow, self).__init__(origin, sample_size)

    @property
    def is_constant(self):
        return False

    def reset_epoch(self):
        super(_EpochResampledDataFlow, self).reset_epoch()
        self._indices = self._make_random_indices(self.sample_size)
        return self
