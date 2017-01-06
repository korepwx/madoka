# -*- coding: utf-8 -*-
import unittest

import numpy as np

from madoka.dataflow import DataFlow
from madoka.dataflow.base import PipelineDataFlow
from madoka.dataflow.in_memory import _NumpyDataFlow
from .helper import (dataflow_slices_generator, dataflow_indices_generator,
                     do_getitem_checks, dataflow_masks_generator,
                     do_subset_all_checks)


class _NoSubsetOptimizeDataFlow(PipelineDataFlow):
    # _NumpyDataFlow might have optimization on `.sliced`, `.indexed`, and
    # `.masked`, thus we use this class to avoid optimization.

    def __init__(self, origin):
        super(_NoSubsetOptimizeDataFlow, self).__init__(origin)

    def all(self):
        return self._origin.all()

    def get(self, item, array_indices=None):
        return self._origin.get(item, array_indices=array_indices)


class SubsetDataFlowTestCase(unittest.TestCase):
    """Unit tests for subset data flows."""

    ARRAY_COUNT = [1, 2]
    ARRAY_LENGTH = [1, 2, 3, 5, 10, 100, 1000]

    def _make_arrays(self):
        for array_count in self.ARRAY_COUNT:
            for length in self.ARRAY_LENGTH:
                yield length, [np.arange(length*(i-1), length*i)
                               for i in range(array_count)]

    def test_sliced(self):
        """Test sliced data flows."""
        for length, arrays in self._make_arrays():
            df = _NoSubsetOptimizeDataFlow(DataFlow.from_numpy(arrays))
            self.assertNotIsInstance(df, _NumpyDataFlow)
            for s in dataflow_slices_generator(length, samples=10):
                s_arrays = [a[s] for a in arrays]
                do_getitem_checks(s_arrays, df[s])
                do_subset_all_checks(s_arrays, df[s])

    def test_indexed(self):
        """Test indexed data flows."""
        for length, arrays in self._make_arrays():
            df = _NoSubsetOptimizeDataFlow(DataFlow.from_numpy(arrays))
            self.assertNotIsInstance(df, _NumpyDataFlow)
            for idx in dataflow_indices_generator(length, samples=25):
                s_arrays = [a[idx] for a in arrays]
                do_getitem_checks(s_arrays, df[idx])
                do_subset_all_checks(s_arrays, df[idx])

    def test_masked(self):
        """Test masked data flows."""
        for length, arrays in self._make_arrays():
            df = _NoSubsetOptimizeDataFlow(DataFlow.from_numpy(arrays))
            self.assertNotIsInstance(df, _NumpyDataFlow)
            for mask in dataflow_masks_generator(length, samples=25):
                s_arrays = [a[mask] for a in arrays]
                do_getitem_checks(s_arrays, df[mask])
                do_subset_all_checks(s_arrays, df[mask])


if __name__ == '__main__':
    unittest.main()
