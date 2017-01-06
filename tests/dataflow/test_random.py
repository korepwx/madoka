# -*- coding: utf-8 -*-
import unittest

import numpy as np

from madoka.dataflow import DataFlow, DataFlowContext


class RandomDataFlowTestCase(unittest.TestCase):
    """Unit tests for random data flows."""

    def test_shuffle_once(self):
        """Test data flows shuffled only once."""
        array = np.arange(1000)
        df = DataFlow.from_numpy(array).shuffle_once()

        # shuffled data flow should be ready after df.shuffle()
        array2 = df.all()[0]
        self.assertFalse(np.all(array2 == array))
        np.testing.assert_array_equal(np.sort(array2), array)

        # after each reset_epoch, the snapshot data flow should be same
        with DataFlowContext(df).as_default() as ctx:
            ctx.reset_epoch()
            array3 = df.all()[0]
            self.assertFalse(np.all(array3 == array))
            np.testing.assert_array_equal(array3, array2)

    def test_shuffle(self):
        """Test data flows shuffled at every epoch."""
        array = np.arange(1000)
        df = DataFlow.from_numpy(array)

        # shuffled data flow should be ready after df.shuffle()
        df2 = df.shuffle()
        array2 = df2.all()[0]
        self.assertFalse(np.all(array2 == array))
        np.testing.assert_array_equal(np.sort(array2), array)

        # after each reset_epoch, the data flow should be reshuffled
        with DataFlowContext(df2).as_default() as ctx:
            ctx.reset_epoch()
            array3 = df2.all()[0]
            self.assertFalse(np.all(array3 == array))
            self.assertFalse(np.all(array3 == array2))
            np.testing.assert_array_equal(np.sort(array3), array)

    def test_resample_once(self):
        """Test data flows re-sampled only once."""
        array = np.arange(1000)
        df = DataFlow.from_numpy(array).resample_once()
        self.assertEquals(len(df), len(array))

        # it should be already re-sampled
        array2 = df.all()[0]
        self.assertFalse(np.all(array2 == array))
        self.assertTrue(np.all(array2 >= 0))
        self.assertTrue(np.all(array2 < len(array)))

        # after each reset_epoch, the snapshot data flow should be same
        with DataFlowContext(df).as_default() as ctx:
            ctx.reset_epoch()
            array3 = df.all()[0]
            self.assertFalse(np.all(array3 == array))
            np.testing.assert_array_equal(array3, array2)

        # test resample with a different size
        df_resize = DataFlow.from_numpy(array).resample_once(10)
        array_resize = df_resize.all()[0]
        self.assertEquals(len(df_resize), 10)
        self.assertTrue(np.all(array_resize >= 0))
        self.assertTrue(np.all(array_resize < len(array)))

    def test_resample(self):
        """Test data flows re-sampled at every epoch."""
        array = np.arange(1000)
        df = DataFlow.from_numpy(array)
        df2 = df.resample()
        self.assertEquals(len(df2), len(df))

        # it should be already re-sampled
        array2 = df2.all()[0]
        self.assertFalse(np.all(array2 == array))
        self.assertTrue(np.all(array2 >= 0))
        self.assertTrue(np.all(array2 < len(array)))

        # after each reset_epoch, the snapshot data flow should not be same
        with DataFlowContext(df2).as_default() as ctx:
            ctx.reset_epoch()
            array3 = df2.all()[0]
            self.assertFalse(np.all(array3 == array))
            self.assertFalse(np.all(array3 == array2))
            self.assertTrue(np.all(array3 >= 0))
            self.assertTrue(np.all(array3 < len(array)))

        # test resample with a different size
        df_resize = DataFlow.from_numpy(array).resample(10)
        array_resize = df_resize.all()[0]
        self.assertEquals(len(df_resize), 10)
        self.assertTrue(np.all(array_resize >= 0))
        self.assertTrue(np.all(array_resize < len(array)))

        with DataFlowContext(df_resize).as_default() as ctx:
            ctx.reset_epoch()
            array_resize2 = df_resize.all()[0]
            self.assertEquals(len(array_resize2), 10)
            self.assertFalse(np.all(array_resize == array_resize2))
            self.assertTrue(np.all(array_resize2 >= 0))
            self.assertTrue(np.all(array_resize2 < len(array)))


if __name__ == '__main__':
    unittest.main()
