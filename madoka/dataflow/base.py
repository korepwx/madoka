# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from madoka.utils import (minibatch_indices_iterator, is_mask_array, where,
                          round_int, split_dataframe_to_arrays)

__all__ = ['DataFlow', 'PipelineDataFlow']


class DataFlow(object):
    """Base interface for data flows.

    A data flow is responsible for reading data from underlying storage.
    All necessary tricks for better performance should be taken care of
    by the data flow, thus the callers do not need to care about how the
    data are actually fetched.
    """

    def __getitem__(self, item):
        if isinstance(item, slice):
            ret = self.sliced(item)
        else:
            if not isinstance(item, np.ndarray):
                item = np.asarray(item)
            if is_mask_array(item):
                ret = self.masked(item)
            else:
                ret = self.indexed(item)
        return ret

    def __len__(self):
        return self.epoch_size

    @property
    def input_flows(self):
        """Get the input data flows.

        Returns
        -------
        tuple[DataFlow]
        """
        raise NotImplementedError()

    @property
    def array_count(self):
        """Count of arrays in this data flow.

        Returns
        -------
        int

        Notes
        -----
        The array count of a flow is guaranteed to be constant.
        """
        raise NotImplementedError()

    @property
    def epoch_size(self):
        """Get the total count of data available in an epoch.

        For data flows with fixed size, this should be the total count of data.
        For real-time generated data flow, this should be a pre-defined number.

        Notes
        -----
        The epoch size of a flow is guaranteed to be constant.
        """
        raise NotImplementedError()

    @property
    def data_shapes(self):
        """Shape of the data, excluding the batch dimension.

        Returns
        -------
        tuple[tuple[int]]
            Shapes of every arrays in the data flow.

        Notes
        -----
        The data shapes of a flow is guaranteed to be constant.
        """
        raise NotImplementedError()

    @property
    def data_types(self):
        """Type of the data elements.

        Returns
        -------
        tuple[np.dtype]
            Data types of every arrays in the data flow.

        Notes
        -----
        The data types of a flow is guaranteed to be constant.
        """
        raise NotImplementedError()

    @property
    def is_constant(self):
        """Whether or not this data flow is constant?

        If a data flow always returns the same set of data in identical order
        for all epochs, then it is regarded as constant.

        Returns
        -------
        bool
        """
        raise NotImplementedError()

    def reset_epoch(self):
        """Reset the internal states for next epoch.

        Calling this method will only reset the states of this data flow.
        The input data flows will not be reset unless the caller manually
        do so.  You may build a ``madoka.dataflow.DataFlowContext`` to
        call ``reset_epoch()`` of all regarding data flows automatically.

        Returns
        -------
        self
        """
        raise NotImplementedError()

    def get(self, item, array_indices=None):
        """Get data at specified indices from the data flow.

        Parameters
        ----------
        item : slice | list | np.ndarray
            Slice, integral indices or boolean masks.

            If indices or masks are specified, the returned data shape will
            be `(indices or masks).shape + data_shape`.

        array_indices : collections.Iterable[int]
            If specified, will return only these arrays.
            If not, will return all arrays.

        Returns
        -------
        tuple[np.ndarray]
        """
        raise NotImplementedError()

    def all(self):
        """Get all data from the data flow.

        Returns
        -------
        tuple[np.ndarray]
        """
        raise NotImplementedError()

    def iter_epoch_batches(self, batch_size, ignore_incomplete_batch=False):
        """Iterate an epoch of the data flow in mini-batches.

        In some ML algorithms, the training data is not consumed as a whole.
        Instead, the training data is consumed by batches with fixed size,
        which is usually known as "mini-batch".

        This method creates a mini-batch iterator, which can be used to
        traverse through a whole epoch.

        Parameters
        ----------
        batch_size : int
            Number of data points in each mini-batch.

        ignore_incomplete_batch : bool
            Whether or not to ignore the final batch if it contains less
            than ``batch-size`` number of items?

        Yields
        ------
        tuple[np.ndarray]
            A mini-batch of this epoch, as a tuple of numpy arrays.

        Notes
        -----
        Furthermore, if the class need to take regard of the tags context,
        it must save the tags as a local variable before the first item has
        been yielded.  (Calling member methods like `.all()` will be safe
        without additional treatment.)
        """
        for indices in minibatch_indices_iterator(
                self.epoch_size, batch_size,
                ignore_incomplete_batch=ignore_incomplete_batch):
            yield self.get(indices)

    def get_array_like(self, array_idx):
        """Get the data array or an array-like proxy at specified index.

        If the data flow is backed by Numpy array, it may return the underlying
        array instead of returning an array-like proxy for best performance.

        Parameters
        ----------
        array_idx : int
            Index of the internal array.

        Returns
        -------
        np.ndarray | madoka.dataflow.arraylike._ArrayLike
            Data array, or array-like proxy of data.
        """
        from .arraylike import _ArrayLike
        return _ArrayLike(self, array_idx)

    def get_numpy_array(self, array_idx):
        """Get the data array at specified index, as numpy ndarray.

        This method will force the data array to be fetched immediately.

        Parameters
        ----------
        array_idx : int
            Index of the internal array.

        Returns
        -------
        np.ndarray
            The data array.
        """
        return self.select_array(array_idx).all()[0]

    def pipeline(self, cls, *args, **kwargs):
        """Apply any pipeline on this data flow.

        Parameters
        ----------
        cls : (DataFlow, *args, **kwargs) -> DataFlow
            The constructor of pipeline data flow.

        *args, **kwargs
            Additional arguments passed to the constructor.

        Returns
        -------
        DataFlow
        """
        return cls(self, *args, **kwargs)

    def select_array(self, array_indices):
        """Select a subset of the arrays to compose a new data flow.

        Parameters
        ----------
        array_indices : int | collections.Iterable[int]
            Indices of the selected arrays.

        Returns
        -------
        DataFlow
        """
        from .subset import _SelectArrayDataFlow
        return self.pipeline(_SelectArrayDataFlow, array_indices)

    def snapshot(self):
        """Build a new data flow by fetching all data from this flow.

        This method fetches all data from this flow immediately, so as to
        construct a new in-memory data flow backed by Numpy array.

        Returns
        -------
        DataFlow
            The snapshot of this data flow.
        """
        from .in_memory import _NumpyDataFlow
        return _NumpyDataFlow(self.all())

    def epoch_cache(self):
        """Build a epoch cached data flow.

        The whole content of this data flow will be re-cached by the returned
        data flow at every epoch.

        Returns
        -------
        DataFlow
        """
        from .cache import _EpochCacheDataFlow
        return self.pipeline(_EpochCacheDataFlow)

    def sliced(self, slice_):
        """Slice this data flow.

        Parameters
        ----------
        slice_ : slice
            Slice instance.

        Returns
        -------
        DataFlow
            The sliced data flow.
        """
        from .subset import _SlicedDataFlow
        return self.pipeline(_SlicedDataFlow, slice_)

    def indexed(self, indices):
        """Get a subset of this data flow according to indices.

        Parameters
        ----------
        indices : np.ndarray
            Integral indices of the subset.

        Returns
        -------
        DataFlow
            The subset data flow.
        """
        from .subset import _IndexedDataFlow
        return self.pipeline(_IndexedDataFlow, indices)

    def masked(self, masks):
        """Get a subset of this data flow according to masks.

        Parameters
        ----------
        masks : np.ndarray
            Boolean masks of the subset.

        Returns
        -------
        DataFlow
            The subset data flow.
        """
        from .subset import _MaskedDataFlow
        return self.pipeline(_MaskedDataFlow, masks)

    def shuffle_once(self):
        """Get a shuffled data flow.

        The returned data flow will be already shuffled, but will not be
        re-shuffled again after calling `reset_epoch()`

        Returns
        -------
        DataFlow
        """
        from .random import _OneTimeShuffledDataFlow
        return self.pipeline(_OneTimeShuffledDataFlow)

    def shuffle(self):
        """Get a shuffled data flow which will be re-shuffled at every epoch.

        The returned data flow will be already shuffled, and will be again
        re-shuffled after calling `reset_epoch()`.

        Returns
        -------
        DataFlow
        """
        from .random import _EpochShuffledDataFlow
        return self.pipeline(_EpochShuffledDataFlow)

    def resample_once(self, sample_size=None):
        """Get a re-sampled data flow.

        The returned data flow will be already re-sampled, but will not
        be re-sampled again after calling `reset_epoch()`.

        Parameters
        ----------
        sample_size : int
            Size of the re-sampled data flow.
            If not specified, will be same as the original data flow.

        Returns
        -------
        DataFlow
        """
        from .random import _OneTimeResampledDataFlow
        return self.pipeline(_OneTimeResampledDataFlow, sample_size)

    def resample(self, sample_size=None):
        """Get a re-sampled data flow which will be re-sampled at every epoch.

        The returned data flow will be already re-sampled, and will be again
        re-sampled after calling `reset_epoch()`.

        Parameters
        ----------
        sample_size : int
            Size of the re-sampled data flow.
            If not specified, will be same as the original data flow.

        Returns
        -------
        DataFlow
        """
        from .random import _EpochResampledDataFlow
        return self.pipeline(_EpochResampledDataFlow, sample_size)

    def apply(self, transformer):
        """Apply the specified transformer on this data flow.

        Parameters
        ----------
        transformer
            A transformer, or a list of transformers to be applied on data.

            If only one transformer is given, it will be applied to the first
            array of this data flow, leaving other arrays unchanged.
            If more than one transformers are given, every one of them will
            be applied to an array of the data flow, at each position.

            The transformers might be a ``madoka.dataflow.Transformer``,
            a scikit-learn transformer (with a `transform` method), any
            callable object with signature `(nd.array) -> nd.array`, or None.
            Specify None for an array will leave it unchanged.

        Returns
        -------
        DataFlow
            The transformed data flow.
        """
        from .transform import _TransformedDataFlow
        return self.pipeline(_TransformedDataFlow, transformer)

    def apply_window(self, window):
        """Apply windows on this data flow, to construct a new flow.

        window : DataWindow | collections.Iterable[DataWindow]
            A window, or a list of windows to be applied on data.

            If only one window is given, it will be applied to the first
            array of this data flow, leaving other arrays unchanged.
            If more than one windows are given, each of them will be applied
            to the array at its position.

            Specify None for an array will leave it unchanged.

        Returns
        -------
        DataFlow
        """
        from .window import _WindowedDataFlow
        return self.pipeline(_WindowedDataFlow, window)

    def split(self, partitions=None, portions=None, shuffle=False):
        """Split this data flow into several partitions.

        One and only one of `partitions` and `portions` should be specified
        in order to determine the size of each partition.  Each of the
        specified `partitions` or `portions` should be positive except one,
        which can be `-1` or `None`, indicating all of the remaining.

        Parameters
        ----------
        partitions : collections.Iterable[int]
            A tuple of integers as the size of each partition.

        portions : collections.Iterable[float]
            A tuple of float numbers as the portion of each partition.
            If `partitions` is specified, this will be ignored.

            It is recommended to use `-1` as one of the portions, otherwise
            the translated partition sizes may not sum up to `epoch_size`.

        shuffle : bool
            Whether or not to shuffle the data flow before splitting?

            If set to True, it is equivalent to `.shuffle_once().
            split(..., shuffle=False)`.

        Raises
        ------
        ValueError
            If the `partitions` cannot sum up to `epoch_size`, or the
            `portions` cannot sum up to 1.

        Returns
        -------
        tuple[DataFlow]
            Splits of the original data flow.
        """
        # check the partitions or portions
        if partitions is None and portions is None:
            raise TypeError('At least one of `partitions` and `portions` '
                            'should be specified.')
        epoch_size = self.epoch_size
        if partitions is None:
            partitions = [
                round_int(p * epoch_size)
                    if p not in (None, -1) and p > 0 else p
                for p in portions
            ]
        if any(p not in (None, -1) and p <= 0 for p in partitions):
            raise ValueError('`partitions` and `portions` must be positive.')
        neg_one_idx = where(p in (None, -1) for p in partitions)
        if len(neg_one_idx) > 1:
            raise ValueError('`-1` is specified for more than once.')
        elif len(neg_one_idx) > 0:
            neg_one_idx = neg_one_idx[0]
            partitions[neg_one_idx] = 0
            remain_size = epoch_size - sum(partitions)
            if remain_size <= 0:
                raise ValueError('Total size of partitions exceeds.')
            partitions[neg_one_idx] = remain_size
        elif sum(partitions) != epoch_size:
            raise ValueError('Total size does not sum up to epoch size.')

        # dealing with shuffle argument
        df = self
        if shuffle:
            df = df.shuffle_once()

        # ready to split the data flow
        ret = []
        start_idx = 0
        for s in partitions:
            ret.append(df[start_idx: start_idx + s])
            start_idx += s
        return tuple(ret)

    def merge(self, *flows):
        """Merge this data flow with other flows.

        Parameters
        ----------
        *flows : tuple[DataFlow | np.ndarray]
            Other data flows.

        Returns
        -------
        DataFlow
        """
        from .merge import merged_data_flow
        return self.pipeline(merged_data_flow, *flows)

    @staticmethod
    def empty():
        """Create an empty data flow.

        Returns
        -------
        DataFlow
        """
        from .empty import empty_flow
        return empty_flow

    @staticmethod
    def from_numpy(arrays):
        """Create a data flow from numpy arrays.

        Parameters
        ----------
        arrays : np.ndarray | collections.Iterable[np.ndarray]
            Numpy array, or a list of arrays.

        Returns
        -------
        DataFlow
        """
        from .in_memory import _NumpyDataFlow
        return _NumpyDataFlow(arrays)

    @staticmethod
    def from_pandas(df, split_label=True, label_as_int=False):
        """Create a data flow from pandas data frame.

        Parameters
        ----------
        df : pd.DataFrame
            Data frame instance.

        split_label : bool
            Whether or not to split "label" column into 2nd array if exists?

        label_as_int : bool
            Whether or not to cast the label array into int32?

        Returns
        -------
        DataFlow
        """
        # convert the data frame to numpy arrays
        if not isinstance(df, pd.DataFrame):
            raise TypeError('`df` must be a pandas DataFrame.')
        # check if the data frame contains NaN values
        if df.isnull().any().any():
            raise ValueError('Unexpected NaN value in data frame.')
        # split the data frame.
        if split_label and 'label' in df:
            arrays = split_dataframe_to_arrays(
                df,
                label_as_int=label_as_int
            )
        else:
            arr = np.asarray(df.values)
            # if there is only one column in the data, we should flatten
            # the array into one dimension.
            if arr.shape[1] == 1:
                arr = arr.reshape((len(arr),))
            arrays = (arr,)
        # now construct the data flow
        from .in_memory import _NumpyDataFlow
        return _NumpyDataFlow(arrays)


class PipelineDataFlow(DataFlow):
    """Base class for all data flows that processes data from another flow.

    Parameters
    ----------
    origin : DataFlow
        Original data flow.
    """

    def __init__(self, origin):
        # memorize the underlying data flow
        self._origin = origin

        # cache properties of the original data flow for faster access
        self._o_array_count = origin.array_count
        self._o_epoch_size = origin.epoch_size
        self._o_data_shapes = origin.data_shapes
        self._o_data_types = origin.data_types
        self._o_is_constant = origin.is_constant

    @property
    def input_flows(self):
        return (self._origin,)

    @property
    def array_count(self):
        return self._o_array_count

    @property
    def epoch_size(self):
        return self._o_epoch_size

    @property
    def data_shapes(self):
        return self._o_data_shapes

    @property
    def data_types(self):
        return self._o_data_types

    @property
    def is_constant(self):
        return self._o_is_constant

    def reset_epoch(self):
        return self
