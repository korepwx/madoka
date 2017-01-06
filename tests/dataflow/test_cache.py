# -*- coding: utf-8 -*-
import unittest

import numpy as np

from madoka.dataflow import DataFlow, DataFlowContext
from .helper import do_getitem_checks, do_subset_all_checks


class CachedDataFlowTestCase(unittest.TestCase):
    """Unit tests for cached data flows."""

    def test_epoch_cached(self):
        array = np.arange(1000)
        df0 = DataFlow.from_numpy(array).shuffle()
        df = df0.epoch_cache()

        # shuffled data with epoch cache will act as if a shuffled data
        array2 = df.all()[0]
        self.assertFalse(np.all(array2 == array))
        np.testing.assert_array_equal(array2, df0.all()[0])

        # test getitem of the cached data flow
        do_getitem_checks(df0.all(), df)
        do_subset_all_checks(df0.all(), df)

        # after each reset_epoch, the data flow should be reshuffled
        with DataFlowContext(df).as_default() as ctx:
            ctx.reset_epoch()
            array3 = df.all()[0]
            self.assertFalse(np.all(array3 == array))
            self.assertFalse(np.all(array3 == array2))
            np.testing.assert_array_equal(array3, df0.all()[0])

        # the content should really be cached
        with DataFlowContext(df0).as_default() as ctx:
            ctx.reset_epoch()
            array4 = df.all()[0]
            self.assertFalse(np.all(array4 == df0.all()[0]))
            np.testing.assert_array_equal(array4, array3)


if __name__ == '__main__':
    unittest.main()
