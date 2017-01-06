# -*- coding: utf-8 -*-
import numpy as np

from madoka.dataflow.in_memory import _NumpyDataFlow
from madoka.utils import unique
from .base import DataFlow
from .empty import _EmptyDataFlow

__all__ = []


def merged_data_flow(*flows):
    """Create a new data flow by merging specified flows.

    If not even one data flow which is not empty has been specified,
    returns an empty data flow.  Futhermore, if the merged data flow
    only contains one data flow, then returns that data flow.

    Otherwise if neither of the above situation happens, returns
    a merged data flow.

    Parameters
    ----------
    *flows : tuple[DataFlow | np.ndarray]
        Other data flows.

    Returns
    -------
    DataFlow
    """
    flows = tuple(f for f in flows if not isinstance(f, _EmptyDataFlow))

    # if no data flows are non-empty, returns an empty data flow
    if not flows:
        return DataFlow.empty()

    # if only one data flow left, returns itself.
    if len(flows) == 1 and isinstance(flows[0], DataFlow):
        return flows[0]

    # otherwise merge these data flows
    ret = _MergedDataFlow(*flows)
    if len(ret.input_flows) == 1:
        # One data flow left after merging, return it.
        # Note this is different from above situation, since multiple
        # numpy arrays may be aggregated into one _NumpyDataFlow.
        return ret.input_flows[0]

    # return the merged data flow.
    return ret


class _MergedDataFlow(DataFlow):
    """Merge data flow.

    This class merges the arrays provided by underlying data flows together,
    to form a new composite data flow.

    Parameters
    ----------
    flows : tuple[DataFlow | np.ndarray]
        The underlying data flows, or numpy arrays which should be
        components of the new data flow.
    """

    def __init__(self, *flows):
        # collect the component data flows
        _flows = []             # type: list[DataFlow]
        _array_buf = []         # type: list[np.ndarray]
        _epoch_size = None      # type: int

        def commit_array_buf():
            if _array_buf:
                _flows.append(DataFlow.from_numpy(_array_buf))
                _array_buf[:] = []

        for i, f in enumerate(flows):
            if isinstance(f, _EmptyDataFlow):
                # skip empty data flows
                continue
            if not isinstance(f, (DataFlow, np.ndarray)):
                raise TypeError(
                    '%r is neither a data flow nor a numpy array.' % (f,))

            # check the epoch size of this component
            if i == 0:
                _epoch_size = len(f)
            elif len(f) != _epoch_size:
                raise TypeError('Array lengths mismatch.')

            # add this component to the flows
            if isinstance(f, _NumpyDataFlow):
                _array_buf.extend(f.numpy_arrays)
            elif isinstance(f, DataFlow):
                commit_array_buf()
                _flows.append(f)
            else:
                _array_buf.append(f)

        commit_array_buf()

        # get the properties of these flows
        if not _flows:
            raise TypeError('No array.')
        _array_count = sum(f.array_count for f in _flows)
        _data_shapes = sum((f.data_shapes for f in _flows), ())
        _data_types = sum((f.data_types for f in _flows), ())
        _is_constant = all(f.is_constant for f in _flows)

        # build the mapping from merged array index to underlying array index
        _mappings = []          # type: list[(int, int)]
        for i, f in enumerate(_flows):
            for j in range(f.array_count):
                _mappings.append((i, j))

        # memorize these objects
        self._flows = tuple(_flows)
        self._array_count = _array_count
        self._epoch_size = _epoch_size
        self._data_shapes = _data_shapes
        self._data_types = _data_types
        self._is_constant = _is_constant
        self._mappings = tuple(_mappings)

    @property
    def input_flows(self):
        return self._flows

    @property
    def array_count(self):
        return self._array_count

    @property
    def epoch_size(self):
        return self._epoch_size

    @property
    def data_shapes(self):
        return self._data_shapes

    @property
    def data_types(self):
        return self._data_types

    @property
    def is_constant(self):
        return self._is_constant

    def reset_epoch(self):
        return self

    def get(self, item, array_indices=None):
        if array_indices is None:
            ret = sum((f.get(item) for f in self._flows), ())
        elif array_indices:
            results = {}
            mapped = [self._mappings[i] for i in array_indices]
            mapped_unique = unique(sorted(mapped))

            # get arrays from each underlying data flow separately
            def commit(left, right):
                flow_idx = mapped_unique[left][0]
                arr_idx = tuple(m[1] for m in mapped_unique[left: right])
                f = self._flows[flow_idx]
                for i, a in enumerate(f.get(item, arr_idx)):
                    results[(flow_idx, arr_idx[i])] = a

            start = 0
            for i in range(1, len(mapped_unique)):
                if mapped_unique[i][0] != mapped_unique[start][0]:
                    commit(start, i)
                    start = i
            commit(start, len(mapped_unique))

            # merge the results
            ret = tuple(results[m] for m in mapped)
        else:
            ret = ()
        return ret

    def all(self):
        return sum((f.all() for f in self._flows), ())
