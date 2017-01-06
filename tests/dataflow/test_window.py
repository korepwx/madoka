# -*- coding: utf-8 -*-
import unittest

import numpy as np

from madoka.dataflow import (NullDataWindow, FixedDataWindow,
                             DailyDigestDataWindow, DataFlow)
from tests.dataflow.helper import do_getitem_checks


class DataWindowTestCase(unittest.TestCase):
    """Unit tests for ``madoka.dataflow.window``."""

    def test_WindowedDataFlow(self):
        """Test windowed data flow."""
        arrays = [
            np.arange(0, 99, 2),
            np.arange(1, 100, 2)
        ]
        df0 = DataFlow.from_numpy(arrays)

        # test with null data window
        w = NullDataWindow()
        flow = df0.apply_window((w, None))
        epoch_tuple = flow.all()
        self.assertEquals(len(epoch_tuple), 2)
        self.assertEquals(len(flow), 50)
        np.testing.assert_array_equal(epoch_tuple[0], arrays[0])
        np.testing.assert_array_equal(epoch_tuple[1], arrays[1])

        # test with a fixed size data window
        w = FixedDataWindow(back_size=2, forward_size=1)
        flow = df0.apply_window(w)
        epoch_tuple = flow.all()
        self.assertEquals(len(epoch_tuple), 2)
        self.assertEquals(len(flow), 47)
        left = np.arange(start=4, stop=98, step=2).reshape((47, 1))
        right = np.arange(start=-4, stop=4, step=2).reshape((1, 4))
        do_getitem_checks([left + right, arrays[1][2: -1]], flow)

    def test_NullDataWindow(self):
        """Test the function of ``NullDataWindow``.

        The most important characteristic of ``NullDataWindow`` is that
        any indices will remain equal after ``batch_index`` is applied.
        Also, the simplicity of ``NullDataWindow`` also allows us to test
        ``IndexableDataWindow`` through ``NullDataWindow``
        """
        w = NullDataWindow()

        # test batch_index method
        indices = np.array(100, dtype=np.int)
        np.testing.assert_array_equal(indices, w.batch_index(indices))
        indices = np.arange(start=0, stop=100, dtype=np.int)
        np.testing.assert_array_equal(indices, w.batch_index(indices))
        indices = indices.reshape((10, 10))
        np.testing.assert_array_equal(indices, w.batch_index(indices))

        # test index method
        self.assertEquals(102, w.index(102))

        # test other members
        self.assertEquals((), w.window_shape)
        self.assertEquals(0, w.look_back_size)
        self.assertEquals(0, w.look_forward_size)

        # test batch_get method, which is inherited from IndexableDataWindow
        # this method is required to produce outputs with the same size as
        # the indices.
        data = np.arange(start=0, stop=100, dtype=np.int)
        indices = np.arange(start=0, stop=100, dtype=np.int)
        np.testing.assert_array_equal(data[indices], w.batch_get(data, indices))
        indices = indices.reshape((10, 10))
        np.testing.assert_array_equal(
            data[indices.flatten()].reshape(indices.shape),
            w.batch_get(data, indices)
        )
        indices = np.array(10, dtype=np.int)
        self.assertEquals(data[10], w.batch_get(data, indices))

    def test_FixedDataWindow(self):
        """Test the function of ``FixedDataWindow``.

        Most of the functions of ``FixedDataWindow`` is inherited from
        ``IndexableDataWindow``, thus need not to be tested.
        """
        # test a normal simple window
        w = FixedDataWindow(back_size=2, forward_size=1)
        self.assertEquals(w.look_back_size, 2)
        self.assertEquals(w.look_forward_size, 1)
        self.assertEquals(w.look_around_size, 3)
        self.assertEquals(w.window_shape, (4,))
        indices = np.arange(start=0, stop=100, dtype=np.int)
        np.testing.assert_array_equal(
            np.arange(start=0, stop=100, dtype=np.int).reshape((100, 1)) +
                np.arange(start=-2, stop=2, dtype=np.int).reshape((1, 4)),
            w.batch_index(indices)
        )

        # test a degenerated simple window
        w = FixedDataWindow(back_size=0, forward_size=0)
        self.assertEquals(w.look_back_size, 0)
        self.assertEquals(w.look_forward_size, 0)
        self.assertEquals(w.look_around_size, 0)
        self.assertEquals(w.window_shape, (1,))
        indices = np.arange(start=0, stop=100, dtype=np.int)
        np.testing.assert_array_equal(
            np.arange(start=0, stop=100, dtype=np.int).reshape((100, 1)),
            w.batch_index(indices)
        )

    def test_DailyDigestDataWindow_aggregation(self):
        """Test the aggregation function from ``DailyDigestDataWindow``."""
        data = np.arange(27, dtype=np.float).reshape((3, 3, 3))

        # test average on 1st and 2nd axis
        w = DailyDigestDataWindow(aggregation='average', time_unit='1min')
        for axis in (0, 1):
            np.testing.assert_array_equal(
                w._build_aggregate_function(axis=axis)(data),
                np.average(data, axis=axis))

        # test median on 1st and 2nd axis
        w = DailyDigestDataWindow(aggregation='median', time_unit='1min')
        for axis in (0, 1):
            np.testing.assert_array_equal(
                w._build_aggregate_function(axis=axis)(data),
                np.median(data, axis=axis))

        # test front sampling on 1st and 2nd axis
        w = DailyDigestDataWindow(aggregation='front_sample', time_unit='1min')
        for axis in (0, 1):
            a = w._build_aggregate_function(axis=axis)(data)
            if axis == 0:
                b = data[0]
            else:
                b = data[[0, 1, 2], 0]
            np.testing.assert_array_equal(a, b)

        # test tail sampling on 1st and 2nd axis
        w = DailyDigestDataWindow(aggregation='tail_sample', time_unit='1min')
        for axis in (0, 1):
            a = w._build_aggregate_function(axis=axis)(data)
            if axis == 0:
                b = data[2]
            else:
                b = data[[0, 1, 2], 2]
            np.testing.assert_array_equal(a, b)

        # test random sampling on 1st and 2nd axis
        def is_random_sample(a, axis):
            if axis == 0:
                return any(np.array_equal(a, data[k]) for k in range(3))
            elif axis == 1:
                for i in range(3):
                    if not any(np.array_equal(a[i], data[i, k])
                               for k in range(3)):
                        return False
            else:
                raise NotImplementedError()
            return True
        w = DailyDigestDataWindow(aggregation='random_sample', time_unit='1min')
        for axis in (0, 1):
            a = w._build_aggregate_function(axis=axis)(data)
            self.assertTrue(
                is_random_sample(a, axis=axis),
                msg='%r is not sampled from %r along axis %r.' %
                    (a, data, axis)
            )

    def test_DailyDigestDataWindow(self):
        """Test the function of ``DailyDigestDataWindow``."""
        def get_epoch(data, window):
            indices = np.arange(window.look_back_size,
                                len(data) - window.look_forward_size)
            return window.batch_get(data, indices)

        DATA_LENGTH = 29 * 24 * 60 + 1
        data = np.arange(DATA_LENGTH)

        # test a daily digest window with all time look back
        w = DailyDigestDataWindow(
            look_back=DailyDigestDataWindow.ALL_TIME_BACK,
            aggregation=DailyDigestDataWindow.AGG_TAIL_SAMPLE,
            time_unit='1min'
        )
        self.assertEquals(w.look_back_size, 29 * 24 * 60)
        self.assertEquals(w.look_forward_size, 0)
        self.assertEquals(w.window_shape, (8, 120))

        generated = get_epoch(data, w)
        answer = np.stack([
            np.arange(DATA_LENGTH-120, DATA_LENGTH),
            np.arange(DATA_LENGTH-120-1440, DATA_LENGTH-1440),
            np.arange(DATA_LENGTH-120-1440*7, DATA_LENGTH-1440*7),
            np.arange(DATA_LENGTH-120-1440*28, DATA_LENGTH-1440*28),
            np.arange(DATA_LENGTH-1429, DATA_LENGTH, step=12),
            np.arange(DATA_LENGTH-1429-1440, DATA_LENGTH-1440, step=12),
            np.arange(DATA_LENGTH-1429-1440*7, DATA_LENGTH-1440*7, step=12),
            np.arange(DATA_LENGTH-1429-1440*28, DATA_LENGTH-1440*28, step=12),
        ])
        np.testing.assert_array_equal(generated, answer.reshape((1, 8, 120)))

        # test a daily digest window with only four weeks look back
        w = DailyDigestDataWindow(
            look_back=DailyDigestDataWindow.FOUR_WEEKS_BACK,
            aggregation=DailyDigestDataWindow.AGG_TAIL_SAMPLE,
            time_unit='1min'
        )
        self.assertEquals(w.look_back_size, 29 * 24 * 60)
        self.assertEquals(w.look_forward_size, 0)
        self.assertEquals(w.window_shape, (4, 120))

        generated = get_epoch(data, w)
        answer = np.stack([
            np.arange(DATA_LENGTH-120, DATA_LENGTH),
            np.arange(DATA_LENGTH-120-1440*28, DATA_LENGTH-1440*28),
            np.arange(DATA_LENGTH-1429, DATA_LENGTH, step=12),
            np.arange(DATA_LENGTH-1429-1440*28, DATA_LENGTH-1440*28, step=12),
        ])
        np.testing.assert_array_equal(generated, answer.reshape((1, 4, 120)))

        # test the window size of other configurations
        w = DailyDigestDataWindow(look_back=DailyDigestDataWindow.ONE_WEEK_BACK,
                                  time_unit='1min')
        self.assertEquals(w.look_back_size, 8 * 24 * 60)
        self.assertEquals(w.window_shape, (4, 120))

        w = DailyDigestDataWindow(look_back=DailyDigestDataWindow.ONE_DAY_BACK,
                                  time_unit='1min')
        self.assertEquals(w.look_back_size, 2 * 24 * 60)
        self.assertEquals(w.window_shape, (4, 120))

if __name__ == '__main__':
    unittest.main()
