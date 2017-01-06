# -*- coding: utf-8 -*-
import unittest

import numpy as np

from madoka.dataflow import DataFlow
from madoka.dataflow.in_memory import _NumpyDataFlow
from .helper import do_getitem_checks, do_subset_all_checks


class MergeTestCase(unittest.TestCase):
    """Test merged data flows."""

    def test_MergedDataFlow(self):
        def f(x):
            # cancel out special optimization on various data flows.
            return x.apply(lambda v: v)

        dataflows = [
            DataFlow.empty(),
            f(DataFlow.from_numpy(np.arange(100, 200, dtype=np.int))),
            np.arange(200, 300, dtype=np.int),
            np.arange(300, 400, dtype=np.int),
            DataFlow.empty(),
            f(DataFlow.from_numpy([np.arange(400, 500, dtype=np.int),
                                   np.arange(500, 600, dtype=np.int)])),
            np.arange(600, 700, dtype=np.int),
            DataFlow.empty()
        ]
        df0 = DataFlow.from_numpy(np.arange(100, dtype=np.int))
        df = df0.merge(*dataflows)

        def check_df():
            # test the basic interfaces of merged data flow
            self.assertEquals(len(df), 100)
            self.assertEquals(df.array_count, 7)
            self.assertEquals(df.data_shapes, ((),) * 7)
            self.assertEquals(df.data_types, (np.int,) * 7)
            self.assertEquals(len(df.input_flows), 5)
            self.assertEquals(df.input_flows[0].array_count, 1)
            self.assertEquals(df.input_flows[1].array_count, 1)
            self.assertEquals(df.input_flows[2].array_count, 2)
            self.assertEquals(df.input_flows[3].array_count, 2)
            self.assertEquals(df.input_flows[4].array_count, 1)

            # test the all results of data flow
            arrays = [np.arange(100, dtype=np.int) + i * 100 for i in range(7)]
            results = df.all()
            self.assertEquals(len(results), 7)
            np.testing.assert_array_equal(arrays, results)

            # test to get partial arrays from data flow
            all_slice = slice(0, None, None)
            for i in range(7):
                np.testing.assert_array_equal(
                    [arrays[i]],
                    df.get(all_slice, array_indices=[i])
                )
                for j in range(7):
                    np.testing.assert_array_equal(
                        [arrays[i], arrays[j]],
                        df.get(all_slice, array_indices=[i, j])
                    )
                    for k in range(7):
                        np.testing.assert_array_equal(
                            [arrays[i], arrays[j], arrays[k]],
                            df.get(all_slice, array_indices=[i, j, k])
                        )

            # regular test on dataflow getitems and subsets
            do_getitem_checks(arrays, df)
            do_subset_all_checks(arrays, df)

        check_df()

        # test to merge the data flows, starting from empty
        df = DataFlow.empty().merge(df0, *dataflows)
        check_df()

    def test_MergeEmptyFlow(self):
        """Test to merge data flow with empty flow envolved."""
        df = DataFlow.from_numpy([np.arange(100)])
        empty = DataFlow.empty()

        # check whether or not two empty data flows are the same instance
        self.assertIs(empty, DataFlow.empty())

        # test to merge nothing, and gets the data flow itself
        self.assertIs(df.merge(), df)

        # test to merge empty, and gets the data flow itself
        self.assertIs(df.merge(empty), df)

        # test to get an emtpy data flow from merge nothing into empty
        self.assertIs(empty.merge(), empty)

        # test to get an empty data flow from merge empty data flow into empty
        self.assertIs(empty.merge(empty), empty)

    def test_MergeArraysAsOneFlow(self):
        """Test to merge numpy arrays into one data flow."""
        df = DataFlow.empty().merge(np.arange(100), np.arange(100, 200))
        self.assertIsInstance(df, _NumpyDataFlow)

if __name__ == '__main__':
    unittest.main()
