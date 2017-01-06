# -*- coding: utf-8 -*-
import numpy as np

from .datatuple import DataTuple

__all__ = [
    'floatX', 'split_dataframe_to_arrays', 'split_numpy_array', 'slice_length',
    'merge_slices', 'merge_slice_indices', 'merge_slice_masks',
    'minibatch_indices_iterator', 'adaptive_density',
]


def floatX(array):
    """Enforce the value type of ``array`` to default float type.

    Parameters
    ----------
    array : numpy.ndarray
        Numpy array, whose value type should be enforced to default
        float type.  If a number is specified, it will be converted
        to 0-dimensional numpy array.

    Returns
    -------
    numpy.ndarray
        The converted numpy array.
    """
    from madoka import config
    return np.asarray(array, dtype=config.floatX)


def split_dataframe_to_arrays(df, label_as_int=False):
    """Split data frame to numpy arrays, with ``label`` column splitted.

    Parameters
    ----------
    df : pandas.DataFrame
        Original data frame, which is expected to contain ``label`` column.

    label_as_int : bool
        If set to True, will discretize the labels into int32, according to
        condition ``labels >= 0.5``.

    Returns
    -------
    DataTuple
        The splitted numpy arrays, where the first represents features without
        labels, and the second represents the labels.
        If the returned features only contain one column, it will be flatten.
    """
    features = df.drop('label', axis=1, inplace=False)
    labels = df['label']
    if label_as_int:
        labels = np.asarray(labels >= 0.5, dtype=np.int32)
    else:
        labels = labels.values
    features = features.values
    if features.shape[1] == 1:
        features = features.flatten()

    return DataTuple(features, labels)


def split_numpy_array(array_or_arrays, right_portion=None, right_size=None,
                      shuffle=True):
    """Split numpy arrays into two halves, by portion or by size.

    Parameters
    ----------
    array_or_arrays : numpy.ndarray | tuple[numpy.ndarray] | DataTuple
        Numpy array, a tuple of numpy arrays, or a ``DataTuple`` instance.

    right_portion : float
        Portion of the right half.  Ignored if ``right_size`` is specified.

    right_size : int
        Size of the right half.

    shuffle : bool
        Whether or not to shuffle before split?

    Returns
    -------
    (numpy.ndarray, numpy.ndarray) | (DataTuple, DataTuple)
        Splitted training and validation data.
        If the given data is a single array, returns the splitted data
        in a tuple of single arrays.  Otherwise a tuple of ``DataTuple``
        instances.
    """
    if isinstance(array_or_arrays, np.ndarray):
        direct_value = True
        data_count = len(array_or_arrays)
    elif isinstance(array_or_arrays, (tuple, list, DataTuple)):
        direct_value = False
        data_count = len(array_or_arrays[0])
    else:
        raise TypeError(
            '%r is neither a numpy array, nor a tuple of arrays.' %
            array_or_arrays
        )

    if right_size is None:
        if right_portion is None:
            raise ValueError('At least one of "right_portion", "right_size" '
                             'should be specified.')

        if right_portion < 0.5:
            right_size = data_count - int(data_count * (1.0 - right_portion))
        else:
            right_size = int(data_count * right_portion)

    if right_size < 0:
        right_size = 0
    if right_size > data_count:
        right_size = data_count

    if shuffle:
        indices = np.arange(data_count)
        np.random.shuffle(indices)
        get_train = lambda v: v[indices[: -right_size]]
        get_valid = lambda v: v[indices[-right_size:]]
    else:
        get_train = lambda v: v[: -right_size, ...]
        get_valid = lambda v: v[-right_size:, ...]

    if direct_value:
        return (get_train(array_or_arrays), get_valid(array_or_arrays))
    else:
        return (DataTuple(*(get_train(v) for v in array_or_arrays)),
                DataTuple(*(get_valid(v) for v in array_or_arrays)))


def _slice_length_sub(start, stop, step):
    if step > 0:
        ret = (stop - start + step - 1) // step
    elif step < 0:
        ret = (start - stop - step - 1) // (-step)
    else:
        raise ValueError('Step of slice cannot be 0.')
    if ret < 0:
        ret = 0
    return ret


def slice_length(length, slice_):
    """Compute the length of slice."""
    start, stop, step = slice_.indices(length)
    return _slice_length_sub(start, stop, step)


def merge_slices(length, *slices):
    """Merge multiple slices into one.

    When we slice some object with the merged slice, it should produce
    exactly the same output as if we slice the object with original
    slices, one after another.  That is to say, if we have:

        merged_slice = merge_slices(len(array), slice1, slice2, ..., sliceN)

    Then these two arrays should be the same:

        array1 = array[slice1][slice2]...[sliceN]
        array2 = array[merged_slice]

    Parameters
    ----------
    length : int
        Length of the array that will be sliced.

    *slices : tuple[slice]
        Sequence of slices to be merged.

    Returns
    -------
    slice
    """
    if isinstance(length, slice):
        raise TypeError('`length` is not specified.')

    # deal with degenerated situations.
    if not slices:
        return slice(0, None, None)
    if len(slices) == 1:
        return slices[0]

    # merge slices
    def merge_two(slice1, slice2):
        start1, stop1, step1 = slice1.indices(length)

        # compute the actual length of slice1
        length1 = _slice_length_sub(start1, stop1, step1)

        # if the length becomes zero after applying slice1, we can stop here
        # by returning an empty slice
        if length1 <= 0:
            return None

        # now, apply slice2 on the slice1
        start2, stop2, step2 = slice2.indices(length1)
        length2 = _slice_length_sub(start2, stop2, step2)
        if length2 <= 0:
            return None

        # shift slice2 by slice1
        step = step1 * step2
        start = start1 + start2 * step1
        assert(0 <= start <= length - 1)

        stop = start1 + stop2 * step1
        assert ((stop - start) * step >= 0)

        if step < 0:
            # special fix: here stop <= -1 indicates to include all data
            #              before start
            if stop <= -1:
                stop = None

        else:
            # special fix: here stop >= n indicates to include all data
            #              after start
            if stop >= length:
                stop = None

        return slice(start, stop, step)

    ret = slices[0]
    for s in slices[1:]:
        ret = merge_two(ret, s)
        if ret is None:
            return slice(0, 0, None)
    return ret


def merge_slice_indices(length, slice_, indices):
    """Merge a slice and integral indices.

    This method merges a slice and integral indices, so that if we have:

        merged = merge_slice_indices(len(array), slice_, indices)

    Then the following two arrays should be same:


        array1 = array[slice_][indices]
        array2 = array[merged]

    Parameters
    ----------
    length : int
        Length of the array that will be indexed.

    slice_ : slice
        Slice of the array.

    indices : np.ndarray
        Indices on the slice of the array.

    Returns
    -------
    np.ndarray
    """
    if not isinstance(indices, np.ndarray):
        indices = np.asarray(indices, dtype=np.int)

    # inspect the slice to get start, stop and step
    start, stop, step = slice_.indices(length)
    assert(0 <= start <= length-1)
    assert(-1 <= stop <= length)
    slen = _slice_length_sub(start, stop, step)

    # merge the slice and the indices
    indices_is_neg = indices < 0
    if np.any(indices_is_neg):
        indices = indices + indices_is_neg * slen
    return start + indices * step


def merge_slice_masks(length, slice_, masks):
    """Merge a slice and boolean masks.

    This method merges a slice and boolean masks, so that if we have:

        merged = merge_slice_masks(len(array), slice_, masks)

    Then the following two arrays should be same:

        array1 = array[slice_][masks]
        array2 = array[merged]

    Parameters
    ----------
    length : int
        Length of the array that will be indexed.

    slice_ : slice
        Slice of the array.

    masks : np.ndarray
        Masks on the slice of the array.

    Returns
    -------
    np.ndarray[int]
        Indices of the chosen elements.
    """
    if len(masks) != slice_length(length, slice_):
        raise TypeError('Length of masks != slice: %r != %r.' %
                        (len(masks), slice_))
    if not isinstance(masks, np.ndarray):
        masks = np.asarray(masks, dtype=np.bool)
    return merge_slice_indices(length, slice_, np.where(masks)[0])


def minibatch_indices_iterator(length, batch_size,
                               ignore_incomplete_batch=False):
    """Iterate through all the mini-batch indices.

    Parameters
    ----------
    length : int
        Total length of data in an epoch.

    batch_size : int
        Size of each mini-batch.

    ignore_incomplete_batch : bool
        Whether or not to ignore the final batch if it contains less
        than ``batch-size`` number of items?

    Yields
    ------
    np.ndarray
        Indices of each mini-batch.  The last mini-batch may contain less
        indices than `batch_size`.
    """
    start = 0
    stop1 = (length // batch_size) * batch_size
    while start < stop1:
        yield np.arange(start=start, stop=start + batch_size, dtype=np.int)
        start += batch_size
    if not ignore_incomplete_batch and start < length:
        yield np.arange(start=start, stop=length, dtype=np.int)


def adaptive_density(values, bins=1000, pscale=10, plimit=.05):
    """Compute the density histogram of given values, with adaptive range.

    It is often the case where outliers which consists only a small portion
    of the data, but expanding the range of data vastly.  This function
    computes the density histogram with adaptive range, so that these outliers
    can be discarded.

    Parameters
    ----------
    values : np.ndarray
        Values whose density should be computed.

    bins : int
        Number of bins of the density.

    pscale : float
        Scaling factor to consider the shrink of density percentile
        having significant effect on shrinking the range of data.

        For example, if `pscale == 10`, then if we can shrink the
        range of data by 10% with only stripping the densities of
        top 0.5% and bottom 0.5%, then we could regard it a significant
        shrinkage.

    plimit : float
        At most this percentile of densities can be shrinked.
        This limit should apply to the sum of top and bottom percentile.

    Returns
    -------
    (np.ndarray, np.ndarray, float, float)
        Density values in each bin, the edges of these bins, as well as
        the stripped densities at the left and the right.  The length of
        edges will be one more than the length of density values.
    """
    hist, edges = np.histogram(values, bins=bins, density=True)
    pwidth = edges[1] - edges[0]
    left = 0
    right = len(hist)
    pleft = 0.0
    pright = 0.0
    candidate = None

    while pleft + pright <= plimit and left < right:
        if pright <= pleft:
            right -= 1
            pright += hist[right] * pwidth
        else:
            pleft += hist[left] * pwidth
            left += 1

        p_data = float(left + len(hist) - right) / len(hist)
        if pleft + pright <= plimit and p_data <= (pleft + pright) * pscale:
            candidate = (left, right)

    if candidate:
        left, right = candidate
        left_stripped = np.sum(hist[:left]) * pwidth
        right_stripped = np.sum(hist[right:]) * pwidth
        hist_sum = np.sum(hist)
        hscale = (hist_sum - (left_stripped + right_stripped)) / hist_sum
        vmin = edges[left]
        vmax = edges[right]
        vstripped = values[(values >= vmin) & (values <= vmax)]
        hist, edges = np.histogram(vstripped, bins=bins, density=True)
        hist *= hscale
    else:
        left_stripped = 0.0
        right_stripped = 0.0

    return hist, edges, left_stripped, right_stripped
