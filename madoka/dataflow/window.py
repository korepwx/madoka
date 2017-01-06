# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import six

from madoka.utils import (merge_slices, merge_slice_indices, merge_slice_masks,
                          is_mask_array, unique, round_int)
from .base import DataFlow, PipelineDataFlow

__all__ = [
    'DataWindow', 'TimeSeriesDataWindow', 'IndexableDataWindow',
    'NullDataWindow', 'FixedDataWindow', 'DailyDigestDataWindow',
]


class _WindowedDataFlow(PipelineDataFlow):
    """Data flow with a window applied.

    Parameters
    ----------
    origin : DataFlow
        Original data flow.

    window : DataWindow | collections.Iterable[DataWindow]
        A window, or a list of windows to be applied on data.

        If only one window is given, it will be applied to the first
        array of this data flow, leaving other arrays unchanged.
        If more than one windows are given, each of them will be applied
        to the array at its position.

        Specify None for an array will leave it unchanged.
    """

    def __init__(self, origin, window):
        null_w = _NullDataWindowSingleton.instance
        if not hasattr(window, '__iter__'):
            windows = (window,) + (null_w,) * (origin.array_count - 1)
        else:
            windows = tuple(null_w if w is None else w for w in window)
        if len(windows) != origin.array_count:
            raise ValueError('Number of windows != array count.')

        super(_WindowedDataFlow, self).__init__(origin)
        self._windows = windows

        # compute the number of elements after window is applied.
        n_backward = max(w.look_back_size for w in self._windows)
        n_forward = max(w.look_forward_size for w in self._windows)
        n_around = n_backward + n_forward
        n_array = self._o_epoch_size
        if n_array <= n_around:
            self._epoch_size = 0
            self._slice = slice(0, 0, None)
        else:
            self._epoch_size = n_array - n_around
            self._slice = slice(n_backward, self._epoch_size + n_backward)

    @property
    def epoch_size(self):
        return self._epoch_size

    @property
    def data_shapes(self):
        return tuple(
            shape if w is None else w.window_shape + shape
            for w, shape in zip(self._windows, self._origin.data_shapes)
        )

    def get(self, item, array_indices=None):
        length = self._o_epoch_size
        if isinstance(item, slice):
            item = merge_slices(length, self._slice, item)
            indices = np.arange(*item.indices(length))
        else:
            if not isinstance(item, np.ndarray):
                item = np.asarray(item)
            if is_mask_array(item):
                indices = merge_slice_masks(length, self._slice, item)
            else:
                indices = merge_slice_indices(length, self._slice, item)
        if array_indices is None:
            array_indices = range(len(self._windows))
        return tuple(
            self._windows[i].batch_get(self._origin.get_array_like(i), indices)
            for i in array_indices
        )

    def all(self):
        return self.get(slice(0, None, None))


class _TimeDeltaConstant:
    """Pre-defined time delta constants."""
    ONE_SECOND = pd.to_timedelta('1s')
    ONE_MINUTE = pd.to_timedelta('1m')
    ONE_DAY = pd.to_timedelta('1d')
    ONE_WEEK = pd.to_timedelta('7d')
    FOUR_WEEKS = pd.to_timedelta('28d')


class DataWindow(object):
    """Base class for all data window strategy.

    A window strategy converts an indexed sequence into a new one,
    such that every element in the new sequence, is composed of
    not only corresponding data point from old sequence,
    but also some additional historical and even future data points,
    which is believed to have impact on that point.
    The element of this sequence is thus called as a "window".

    Notes
    -----
    Window strategy will only be applied to the 1st dimension.
    All remaining dimensions will be copied to the output sequence
    without further processing.
    """

    def get(self, array, index):
        """Get the window of one data point at specified index.

        Parameters
        ----------
        array : np.ndarray | ArrayLike
            The original sequence data.

        index : int
            The 1st dimensional index of the point in original data.

        Returns
        -------
        np.ndarray
            The window for specified point.
        """
        return self.batch_get(array, [index])[0]

    def batch_get(self, array, indices):
        """Get windows of multiple points at specified indices.

        Parameters
        ----------
        array : np.ndarray | ArrayLike
            The original sequence data.

        indices : collections.Iterable[int]
            The 1st dimensional indices of the points in original data.

        Returns
        -------
        np.ndarray
            The windows for specified points.
        """
        raise NotImplementedError()

    @property
    def window_shape(self):
        """Get the shape of each window.

        Returns
        -------
        tuple[int]
            Shape of the window, as a tuple of integers.

        Notes
        -----
        The shape of window might be an empty tuple, if the output sequence
        is chosen to be exactly the original time series data.
        """
        raise NotImplementedError()

    @property
    def look_back_size(self):
        """Get the look back size for each window.

        A window strategy might need to look back to historical data points
        before it can compute the window for a particular data point.
        This property should indicate the maximum gap between the considered
        data point, and the furthest required historical point.

        For example, if we have three points `p1, p2, p3` in a row, then
        the ``look_back_size`` is 1 if the window for `p3` depends only
        on `p2`, and it should be 2 if `p3` depends on `p1` and `p2`,
        or even depends only on `p1`.

        Returns
        -------
        int
        """
        raise NotImplementedError()

    @property
    def look_forward_size(self):
        """Get the look forward size for each window.

        A window strategy might need to look forward to future data points
        before it can compute the window for a particular data point.
        This property should indicate the maximum gap between the considered
        data point, and the furthest required future point.

        Returns
        -------
        int
        """
        raise NotImplementedError()

    @property
    def look_around_size(self):
        """Get the look around size for each window.

        Look around size = look back size + look forward size.

        Returns
        -------
        int
        """
        return self.look_back_size + self.look_forward_size


class IndexableDataWindow(DataWindow):
    """Base class for window strategy that only copies original data.

    Some window strategies produces windows only by copying data points
    from certain indices in the original data.  They do not transform
    the data, so it allows the caller to inspect the window strategy
    in detail, by getting the actual data indices for a certain window.
    """

    def index(self, index):
        """Get the point indices composing the window of one point.

        Parameters
        ----------
        index : int
            The 1st dimensional index of the point in original data.

        Returns
        -------
        np.ndarray
            The indices of points composing the window.
        """
        return self.batch_index([index])[0]

    def batch_index(self, indices):
        """Get the point indices composing the windows for multiple points.

        Parameters
        ----------
        indices : collections.Iterable[int]
            The 1st dimensional indices of the points in original data.

        Returns
        -------
        np.ndarray
            The indices of points composing the window.
        """
        raise NotImplementedError()

    def batch_get(self, array, indices):
        i = self.batch_index(indices)
        return array[i.flatten()].reshape(i.shape + array.shape[1:])


class TimeSeriesDataWindow(DataWindow):
    """Base class for window strategy for time series data.

    A time series data window might not operate on absolute indices.
    Instead it may take advantage of the datetime index to produce
    data windows.

    Parameters
    ----------
    time_unit : pandas.Timedelta | numpy.timedelta64 | str
        Time unit of the data.

    time_index : pandas.DateTimeIndex
        The datetime index of the data.  Datetime index must have
        at least two elements, and the intervals of the index must
        be homogeneous.
    """

    def __init__(self, time_unit=None, time_index=None):
        self._time_unit = None
        self._time_index = None
        if time_index is not None:
            self.set_time_index(time_index)
        elif time_unit is not None:
            self.set_time_unit(time_unit)
        else:
            raise TypeError('At least one of `time_unit` and `time_index` '
                            'should be specified.')

    @property
    def time_index(self):
        """Get the datetime index of the data."""
        return self._time_index

    def set_time_index(self, value):
        """Set the datetime index of the data.

        Derived classes should override this to refresh internal statuses
        when a different time index instance is assigned.

        Parameters
        ----------
        value : pandas.DateTimeIndex
            The datetime index of the data.  Datetime index must have
            at least two elements, and the intervals of the index must
            be homogeneous.

        Raises
        ------
        ValueError
            If the assigned time index is not homogeneous.

        Notes
        -----
        This will also set the time unit of data.
        """
        if value is not None:
            # check the time intervals
            time_ticks = value.values
            if len(time_ticks) < 2:
                raise ValueError('There must be at least two time ticks.')
            intervals = np.unique(time_ticks[1:] - time_ticks[:-1])
            if len(intervals) > 1:
                raise ValueError('Time intervals are not homogeneous.')

            # we must set the time unit before setting time index,
            # to allow ``set_time_unit`` break the whole process
            # by raising an error.
            self.set_time_unit(value[1] - value[0])
            self._time_index = value
        else:
            self.set_time_unit(None)
            self._time_index = None

    @property
    def time_unit(self):
        """Get the time unit of the data."""
        return self._time_unit

    def set_time_unit(self, value):
        """Set the time unit of the data.

        Derived classes should override this to refresh internal statuses
        when a different time unit is assigned.

        Parameters
        ----------
        value : pandas.Timedelta | numpy.timedelta64 | str
            Time unit of the data.
        """
        if value is not None and not isinstance(value, pd.Timedelta):
            value = pd.to_timedelta(value)
        self._time_unit = value


class NullDataWindow(IndexableDataWindow):
    """Null data window strategy, which outputs exactly the input sequence."""

    def batch_index(self, indices):
        return np.asarray(indices, dtype=np.int)

    @property
    def window_shape(self):
        return ()

    @property
    def look_back_size(self):
        return 0

    @property
    def look_forward_size(self):
        return 0


class _NullDataWindowSingleton:
    instance = NullDataWindow()


class FixedDataWindow(IndexableDataWindow):
    """Window strategy which looks back and forward for a fixed size.

    This fixed window strategy just looks back and forward, to collect a
    continuous block with fixed size.  The size of each window should then
    be `look_back_size + 1 + look_forward_size`.

    Parameters
    ----------
    back_size : int
        Look back size for this window.

    forward_size : int
        Look forward size for this window.

    exclude_self : bool
        Whether or not to exclude the self point of each window?

        If the self point is excluded, the size of each window should be
        `look_back_size + look_forward_size`.

    Notes
    -----
    When `look_back_size == look_forward_size == 0`, this window strategy
    is still not equivalent to ``NullDataWindow``, since it will have
    an additional dimension for each window in the generated sequence.
    """

    def __init__(self, back_size, forward_size, exclude_self=False):
        if not back_size and not forward_size and exclude_self:
            raise TypeError('Empty data window.')
        super(FixedDataWindow, self).__init__()
        self._back_size = back_size
        self._forward_size = forward_size
        self._exclude_self = exclude_self

    def batch_index(self, indices):
        if self._exclude_self:
            window_biases = np.arange(
                start=-self.look_back_size,
                stop=self.look_forward_size,
                dtype=np.int
            )
            if self.look_forward_size > 0:
                window_biases[:, -self.look_forward_size:] += 1
        else:
            window_biases = np.arange(
                start=-self.look_back_size,
                stop=self.look_forward_size+1,
                dtype=np.int
            )
        ret_shape = (1,) * len(indices.shape) + self.window_shape
        window_biases = window_biases.reshape(ret_shape)
        indices_extended = indices.reshape(indices.shape + (1,))
        return np.asarray(window_biases + indices_extended, dtype=np.int)

    @property
    def window_shape(self):
        if self._exclude_self:
            shape = (self.look_around_size, )
        else:
            shape = (self.look_around_size + 1, )
        return shape

    @property
    def look_back_size(self):
        return self._back_size

    @property
    def look_forward_size(self):
        return self._forward_size


class DailyDigestDataWindow(TimeSeriesDataWindow):
    """Data window strategy that computes daily digests.

    A daily digest is an overall image of what the time series looks like
    within one day nearby a given point.  It is by default a two dimensional
    array, in the shape (3, 120) or (2, 120), according to the time unit of
    the original data.

    If the time unit of original data is `1s`, then the shape of daily digest
    should be (3, 120), where the 1st row should be the exact values of data
    within the last two minutes.  The 2nd row indicates an aggregation over
    the last two hours, thus each element of this row should be an aggregation
    over one minute.  The 3rd row represents an aggregation over the day,
    with each element corresponds to 12 minutes.

    The daily digest is very similar to above mentioned when the time unit
    is `1min`.  The only difference is that the first row does not exist.
    Any other time unit will be refused by this window strategy.

    The way to do aggregation is controlled by ``aggregation`` argument.
    Besides, it is possible to have a larger window, by taking the daily
    digest one day, one week, one_month, or even one year ago.  This can
    be controlled by ``look_back`` argument.

    It is possible to take more historical into account, by computing the
    daily digest for not only the day, but also the last day, the day one
    week ago, and the day four weeks ago.  These four digests should be
    gathered together in a specially designed way: the first rows of the
    four digests are gathered together to form the first four rows of the
    resulted digest, and so are the second and third rows.  This can help
    CNN to capture relationship between the four digests with its local
    receptive field.

    Parameters
    ----------
    look_back : str | collections.Iterable[str]
        Set of strings to indicate the look back strategy.
        Possible choices are:

        *  ``ONE_DAY_BACK``: take digest of last day.
        *  ``ONE_WEEK_BACK``: take digest of the day one week ago.
        *  ``FOUR_WEEKS_BACK``: take digest of the day four weeks ago.

        You may specify ``ALL_TIME_BACK`` to include all of these choices.

    aggregation : str
        The aggregation method.  Possible choices are:

        *  ``AGG_AVERAGE``: take the average over a period of time.
        *  ``AGG_MEDIAN``: take the median over a period of time.
        *  ``AGG_FRONT_SAMPLE``: take the first element in each time period.
        *  ``AGG_TAIL_SAMPLE``: take the last element in each time period.
        *  ``AGG_RANDOM_SAMPLE``: take a random element in each time period.

    split_channel : bool
        If set to True, will split samples of different time units into
        different channels.
    """

    # look back strategy
    ONE_DAY_BACK = '1 day'
    ONE_WEEK_BACK = '1 week'
    FOUR_WEEKS_BACK = '4 weeks'

    # full set of the look back strategies
    ALL_TIME_BACK = (ONE_DAY_BACK, ONE_WEEK_BACK, FOUR_WEEKS_BACK)

    # aggregation strategy
    AGG_AVERAGE = 'average'
    AGG_MEDIAN = 'median'
    AGG_FRONT_SAMPLE = 'front_sample'
    AGG_TAIL_SAMPLE = 'tail_sample'
    AGG_RANDOM_SAMPLE = 'random_sample'

    # full set of aggregation methods
    ALL_AGGREGATION_METHODS = (AGG_AVERAGE, AGG_MEDIAN, AGG_FRONT_SAMPLE,
                               AGG_TAIL_SAMPLE, AGG_RANDOM_SAMPLE)

    def __init__(self, look_back=ALL_TIME_BACK, aggregation=AGG_AVERAGE,
                 split_channel=False, time_unit=None, time_index=None):
        super(DailyDigestDataWindow, self).__init__(time_unit=time_unit,
                                                    time_index=time_index)

        # parse the look back strategy
        if isinstance(look_back, six.string_types):
            look_back = [look_back]
        look_back = unique(look_back)
        look_back_flag = 0
        for s in look_back:
            if s not in self.ALL_TIME_BACK:
                raise ValueError('Unknown look back strategy %r.' % s)
            # convert the string flags into integral masks.
            idx = self.ALL_TIME_BACK.index(s)
            look_back_flag |= (1 << idx)
        self._look_back_flag = look_back_flag

        # parse the aggregation method
        if aggregation not in self.ALL_AGGREGATION_METHODS:
            raise ValueError('Unknown aggregation method %r.' % aggregation)
        self.aggregation = aggregation

        # memorize split channels option
        self.split_channel = split_channel

    def set_time_unit(self, value):
        # compute important properties
        if value is not None:
            if not isinstance(value, pd.Timedelta):
                value = pd.to_timedelta(value)
            if value not in (_TimeDeltaConstant.ONE_SECOND,
                             _TimeDeltaConstant.ONE_MINUTE):
                raise ValueError('Time unit %r is not supported.' % value)
            # pre-compute the length of data that spans one day
            time_unit_is_second = round_int(value.total_seconds()) == 1
            day_span = round_int(
                _TimeDeltaConstant.ONE_DAY.total_seconds() /
                value.total_seconds()
            )
        else:
            time_unit_is_second = day_span = None

        # now actually assign to the properties
        super(DailyDigestDataWindow, self).set_time_unit(value)
        self._time_unit_is_second = time_unit_is_second
        self._day_span = day_span

    @property
    def window_shape(self):
        digest_size = 2 + self._time_unit_is_second
        digest_count = (
            1  # current day
            + (self._look_back_flag & 0x1)  # one day ago
            + ((self._look_back_flag >> 1) & 0x1)  # one week ago
            + ((self._look_back_flag >> 2) & 0x1)  # four weeks ago
        )
        if self.split_channel:
            return (digest_count, 120, digest_size)
        else:
            return (digest_size * digest_count, 120)

    @property
    def look_back_size(self):
        if 0x4 & self._look_back_flag:
            return self._day_span * 29
        elif 0x2 & self._look_back_flag:
            return self._day_span * 8
        elif 0x1 & self._look_back_flag:
            return self._day_span * 2
        else:
            return self._day_span

    @property
    def look_forward_size(self):
        return 0

    def _build_aggregate_function(self, axis):
        """Build aggregation function for specified axis.
        Note that the regarding axis will be reduced after aggregation.
        """
        if axis == 0:
            # the axis to be operated is exactly the first dimension,
            # which makes it very easy to do random sampling.
            # we just choose a random index, and then select that element.
            def random_agg(a):
                return a[np.random.randint(0, a.shape[0], dtype=np.int)]
        elif axis == 1:
            # the axis to be operated is exactly the second dimension,
            # thus we should select on this axis.
            def random_agg(a):
                selector0 = np.arange(a.shape[0], dtype=np.int)
                selector1 = np.random.randint(0, a.shape[1], (a.shape[0],),
                                              dtype=np.int)
                return a[selector0, selector1]
        else:
            raise NotImplementedError()

        agg_func = {
            self.AGG_AVERAGE: lambda a: np.average(a, axis=axis),
            self.AGG_MEDIAN: lambda a: np.median(a, axis=axis),
            self.AGG_FRONT_SAMPLE: lambda a: np.take(a, 0, axis=axis),
            self.AGG_TAIL_SAMPLE: lambda a: np.take(a, -1, axis=axis),
            self.AGG_RANDOM_SAMPLE: random_agg,
        }
        return agg_func[self.aggregation]

    def _build_digest_function(self, array):
        """Build function that computes digest for specified index.

        Parameters
        ----------
        array : np.ndarray
            Time series data.
        """
        def f(arr, head_shape):
            return agg_func(arr.reshape(head_shape + arr.shape[1:]))

        # build aggregation function
        agg_func = self._build_aggregate_function(axis=1)

        # compute the index offsets for digest
        offsets = [0]
        if 0x1 & self._look_back_flag:
            offsets.append(
                round_int(_TimeDeltaConstant.ONE_DAY.total_seconds()))
        if 0x2 & self._look_back_flag:
            offsets.append(
                round_int(_TimeDeltaConstant.ONE_WEEK.total_seconds()))
        if 0x4 & self._look_back_flag:
            offsets.append(
                round_int(_TimeDeltaConstant.FOUR_WEEKS.total_seconds()))
        offsets = np.asarray(offsets, dtype=np.int)

        # build daily digest pieces generator
        if self._time_unit_is_second:
            channels = 3

            def make_pieces(index):
                return (
                    # values for the two minutes
                    tuple(array[index-offset-120+1: index-offset+1]
                          for offset in offsets),
                    # values for the two hours
                    tuple(f(array[index-offset-7200+1: index-offset+1],
                            (120, 60))
                          for offset in offsets),
                    # values for the whole day
                    tuple(f(array[index-offset-86400+1: index-offset+1],
                            (120, 720))
                          for offset in offsets)
                )
        else:
            channels = 2
            offsets = np.asarray(offsets / 60, dtype=np.int)

            def make_pieces(index):
                return (
                    # array slice for the two hours
                    tuple(array[index-offset-120+1: index-offset+1]
                          for offset in offsets),
                    # array slice for the whole day
                    tuple(f(array[index-offset-1440+1: index-offset+1],
                            (120, 12))
                          for offset in offsets)
                )

        # build the final digest generator
        if self.split_channel:
            def func(index):
                pieces = make_pieces(index)
                x = np.asarray(pieces, dtype=array.dtype)
                x = x.reshape((channels, -1))
                x = x.transpose().reshape((-1, 120, channels))
                return x
        else:
            def func(index):
                pieces = make_pieces(index)
                ret = np.asarray(pieces, dtype=array.dtype).reshape((-1, 120))
                return ret
        return func

    def batch_get(self, array, indices):
        # first, build the slicing function and the aggregation function
        indices_shape = indices.shape
        indices = indices.flatten()
        digest_func = self._build_digest_function(array)

        # next, gather all digest on each index together.
        digested = np.stack([digest_func(i) for i in indices])
        return digested.reshape(indices_shape + digested.shape[1:])
