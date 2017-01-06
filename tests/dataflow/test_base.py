# -*- coding: utf-8 -*-
import unittest

import numpy as np

from madoka.dataflow import DataFlow, DataFlowContext


class BaseDataFlowTestCase(unittest.TestCase):
    """Unit tests for base data flows."""

    def test_snapshot(self):
        """Test snapshot data flows."""
        array = np.arange(1000)
        df = DataFlow.from_numpy(array).shuffle().snapshot()
        with DataFlowContext(df).as_default() as ctx:
            # shuffled data flow should be ready after df.shuffle()
            array2 = df.all()[0]
            self.assertFalse(np.all(array2 == array))
            np.testing.assert_array_equal(np.sort(array2), array)

            # after each reset_epoch, the snapshot data flow should be same
            ctx.reset_epoch()
            array3 = df.all()[0]
            self.assertFalse(np.all(array3 == array))
            np.testing.assert_array_equal(array3, array2)

    def test_split(self):
        """Test split data flows."""
        array = np.arange(1000)
        df = DataFlow.from_numpy(array)

        # first, test throw errors on invalid arguments
        def assert_invalid_arg(**kwargs):
            with self.assertRaises(ValueError):
                df.split(**kwargs)
        assert_invalid_arg(partitions=[])
        assert_invalid_arg(partitions=[1000, 1])
        assert_invalid_arg(partitions=[1000, -1])
        assert_invalid_arg(partitions=[1, 2])
        assert_invalid_arg(portions=[])
        assert_invalid_arg(portions=[1.0, 0.1])
        assert_invalid_arg(portions=[1.0, -1])
        assert_invalid_arg(portions=[0.1, 0.2])

        # next, test split without shuffling
        df1, df2, df3 = df.split(partitions=[700, 200, 100])
        np.testing.assert_array_equal(df1.all()[0], array[:700])
        np.testing.assert_array_equal(df2.all()[0], array[700:900])
        np.testing.assert_array_equal(df3.all()[0], array[900:1000])
        df1, df2, df3 = df.split(portions=[-1, 0.2, 0.1])
        np.testing.assert_array_equal(df1.all()[0], array[:700])
        np.testing.assert_array_equal(df2.all()[0], array[700:900])
        np.testing.assert_array_equal(df3.all()[0], array[900:1000])

        # finally, test split with shuffling
        df1, df2 = df.split(portions=[0.5, -1], shuffle=True)
        self.assertEquals(len(df1), 500)
        self.assertEquals(len(df2), 500)
        df_array = np.concatenate([df1.all()[0], df2.all()[0]], axis=0)
        self.assertFalse(np.all(df_array == array))
        np.testing.assert_array_equal(np.sort(df_array), array)

if __name__ == '__main__':
    unittest.main()
