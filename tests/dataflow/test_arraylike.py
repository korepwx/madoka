# -*- coding: utf-8 -*-
import unittest

import numpy as np

from madoka.dataflow import DataFlow
from madoka.dataflow.arraylike import _ArrayLike
from .helper import (dataflow_slices_generator, dataflow_indices_generator,
                     dataflow_masks_generator, format_slice)


class ArrayLikeTestCase(unittest.TestCase):
    """Unit tests for `madoka.dataflow.ArrayLike`."""
    
    def test_basic_functions(self):
        """Test the basic functions of ArrayLike object."""
        def test_getitem(array, npy_array, item):
            try:
                np.testing.assert_array_equal(array[item], npy_array[item])
            except AssertionError as ex:
                if isinstance(item, slice):
                    item_str = format_slice(item)
                else:
                    item_str = str(list(item))
                array_str = str(list(npy_array))
                msg = ('getitem failed:\n  array: %s\n  item: [%s]' %
                       (array_str, item_str))
                msg += ex.args[0]
                ex.args = (msg,) + ex.args[1:]
                raise ex

        def do_test(arrays):
            df = DataFlow.from_numpy(arrays)
            self.assertEquals(df.array_count, len(arrays))
            for i in range(df.array_count):
                array = _ArrayLike(df, i)
                self.assertEquals(len(array), len(arrays[i]))
                # test `__iter__` of this array-like object
                np.testing.assert_array_equal(np.asarray(array), arrays[i])
                # test `to_numpy` of this array-like object
                np.testing.assert_array_equal(array.to_numpy(), arrays[i])
                # test `__getitem__` of the array
                for s in dataflow_slices_generator(len(array)):
                    test_getitem(array, arrays[i], s)
                for idx in dataflow_indices_generator(len(array)):
                    test_getitem(array, arrays[i], idx)
                for mask in dataflow_masks_generator(len(array)):
                    test_getitem(array, arrays[i], mask)

        # first test, length % DEFAULT_PREDICT_BATCH_SIZE != 0
        do_test([np.arange(2000), np.arange(2000, 4000)])
        # second test, length % DEFAULT_PREDICT_BATCH_SIZE == 0
        do_test([np.arange(2048), np.arange(2048, 4096)])


if __name__ == '__main__':
    unittest.main()
