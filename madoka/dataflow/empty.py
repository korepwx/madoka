# -*- coding: utf-8 -*-
import numpy as np

from .base import DataFlow

__all__ = []


class _EmptyDataFlow(DataFlow):
    """Empty data flow.

    This class is useful when constructing a data flow by merging
    sources from other data flows.
    """

    @property
    def input_flows(self):
        return ()

    @property
    def array_count(self):
        return 0

    @property
    def epoch_size(self):
        return 0

    @property
    def data_shapes(self):
        return ()

    @property
    def data_types(self):
        return ()

    @property
    def is_constant(self):
        return True

    def reset_epoch(self):
        pass

    def get(self, item, array_indices=None):
        if array_indices:
            raise IndexError('Empty flow does not contain any array.')
        return ()

    def all(self):
        return ()

    def iter_epoch_batches(self, batch_size, ignore_incomplete_batch=False):
        return iter(())

    def get_array_like(self, array_idx):
        raise IndexError('Empty flow does not contain any array.')

    def pipeline(self, cls, *args, **kwargs):
        # Overriding pipeline will effectively override almost
        # all the transformation methods.
        return self

    def select_array(self, array_indices):
        if array_indices:
            raise IndexError('Empty flow does not contain any array.')
        return self

    def _get_item_on_empty_flow(self, slice_indices_or_mask):
        try:
            np.asarray([])[slice_indices_or_mask]
        except IndexError:
            raise IndexError('Empty flow does not contain any data.')
        else:
            return self

    def sliced(self, slice_):
        return self._get_item_on_empty_flow(slice_)

    def indexed(self, indices):
        return self._get_item_on_empty_flow(indices)

    def masked(self, masks):
        return self._get_item_on_empty_flow(masks)

    def merge(self, *flows):
        from .merge import merged_data_flow
        return merged_data_flow(*flows)

# use a global shared empty data flow.
empty_flow = _EmptyDataFlow()
