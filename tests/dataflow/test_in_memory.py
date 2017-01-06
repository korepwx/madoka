# -*- coding: utf-8 -*-
import unittest

import numpy as np

from madoka.dataflow import DataFlow
from madoka.dataflow.in_memory import _NumpyDataFlow
from tests.dataflow.helper import do_getitem_checks, do_subset_all_checks


class InMemoryDataFlowTestCase(unittest.TestCase):
    """Unit tests for in-memory data flows."""

    def test_numpy(self):
        """Test the functions of Numpy data flow."""
        def do_test(arrays):
            df = DataFlow.from_numpy(arrays)

            # check whether or not numpy data flow has optimized subsets
            self.assertIsInstance(df[:], _NumpyDataFlow)
            self.assertIsInstance(df[[0, 1, 2]], _NumpyDataFlow)
            self.assertIsInstance(
                df[np.random.binomial(1, .5, size=len(df)).astype(np.bool)],
                _NumpyDataFlow
            )

            # check get item on numpy data flow
            do_getitem_checks(arrays, df)
            do_subset_all_checks(arrays, df)

        do_test([np.arange(100)])
        do_test([np.arange(128)])
        do_test([np.arange(16).reshape(4, 4)])
        do_test([np.arange(16).reshape(4, 4), np.arange(16).reshape(4, 4) + 16])

if __name__ == '__main__':
    unittest.main()
