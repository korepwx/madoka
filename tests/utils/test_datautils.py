# -*- coding: utf-8 -*-
import itertools
import unittest

import numpy as np

from madoka.utils import (slice_length, merge_slices, merge_slice_indices,
                          merge_slice_masks, DEFAULT_PREDICT_BATCH_SIZE,
                          adaptive_density)
from tests.dataflow.helper import (format_slice, dataflow_slices_generator,
                                   dataflow_indices_generator,
                                   dataflow_masks_generator)


class DataUtilsTestCase(unittest.TestCase):
    """Unit tests for ``madoka.utils.datautils``."""

    ARRAY_LENGTHS = [1, 2, 3, 4, 5, 10,
                     DEFAULT_PREDICT_BATCH_SIZE-1, DEFAULT_PREDICT_BATCH_SIZE,
                     DEFAULT_PREDICT_BATCH_SIZE+1]

    def test_slice_length(self):
        """Test ``slice_length`` function."""
        for length in self.ARRAY_LENGTHS:
            data = np.arange(length)
            slices = dataflow_slices_generator(length, samples=50)
            for s in slices:
                slen = slice_length(len(data), s)
                answer = len(data[s])
                self.assertEquals(
                    slen, answer,
                    msg='len(%r[%s]) is %r rather than %r' %
                        (data, format_slice(s), answer, slen)
                )

    @staticmethod
    def _add_merge_message(ex, data, s1, s2, merged):
        data_str = str(list(data))
        merged_str = '?' \
            if not isinstance(merged, slice) else format_slice(merged)
        s2_str = format_slice(s2) if isinstance(s2, str) else str(list(s2))
        msg = ('%s[%s] != %s[%s][%s]' %
               (data_str, merged_str, data_str, format_slice(s1), s2_str))
        ex.args = (msg + '\n  ' + ex.args[0],) + ex.args[1:]
        return ex

    def test_merge_slices(self):
        """Test ``merge_slices`` function."""
        for length in self.ARRAY_LENGTHS:
            data = np.arange(length)
            slices = list(dataflow_slices_generator(length, samples=50))
            for s1, s2 in itertools.product(slices, slices):
                merged = '?'
                try:
                    merged = merge_slices(length, s1, s2)
                    np.testing.assert_array_equal(data[merged], data[s1][s2])
                except AssertionError as ex:
                    raise self._add_merge_message(ex, data, s1, s2, merged)

    def test_merge_slice_indices(self):
        """Test ``merge_slice_indices`` function."""
        for length in self.ARRAY_LENGTHS:
            slices = list(dataflow_slices_generator(length, samples=50))
            for s in slices:
                slen = slice_length(length, s)
                if slen > 0:
                    data = np.arange(length)
                    indices = list(dataflow_indices_generator(slen, samples=100))
                    for idx in indices:
                        merged = '?'
                        try:
                            merged = merge_slice_indices(length, s, idx)
                            np.testing.assert_array_equal(data[merged],
                                                          data[s][idx])
                        except Exception as ex:
                            raise self._add_merge_message(ex, data, s, idx,
                                                          merged)

    def test_merge_slice_masks(self):
        """Test ``merge_slice_masks`` function."""
        for length in self.ARRAY_LENGTHS:
            slices = list(dataflow_slices_generator(length, samples=50))
            for s in slices:
                slen = slice_length(length, s)
                if slen > 0:
                    data = np.arange(length)
                    indices = list(dataflow_masks_generator(slen, samples=100))
                    for idx in indices:
                        merged = '?'
                        try:
                            merged = merge_slice_masks(length, s, idx)
                            np.testing.assert_array_equal(data[merged],
                                                          data[s][idx])
                        except Exception as ex:
                            raise self._add_merge_message(ex, data, s, idx,
                                                          merged)


if __name__ == '__main__':
    unittest.main()
