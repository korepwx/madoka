# -*- coding: utf-8 -*-
import numpy as np

from .base import DataFlow, PipelineDataFlow
from .helper import detect_output_shape_dtype, apply_function_on_array

__all__ = ['Transformer']


class _TransformedDataFlow(PipelineDataFlow):
    """Data flow transformed from another data flow.

    Parameters
    ----------
    origin : DataFlow
        Original data flow.

    transformer
        A transformer, or a list of transformers to be applied on data.

        If only one transformer is given, it will be applied to the first
        array of this data flow, leaving other arrays unchanged.
        If more than one transformers are given, each of them will be applied
        to the array at its position.

        The transformers might be a ``madoka.dataflow.Transformer``,
        a scikit-learn transformer (with a `transform` method), any
        callable object with signature `(nd.array) -> nd.array`, or None.
        Specify None for an array will leave it unchanged.
    """

    def __init__(self, origin, transformer):
        super(_TransformedDataFlow, self).__init__(origin)

        def to_transformer(t):
            if t is None:
                t = _DummyTransformerSingleton.instance
            elif not isinstance(t, Transformer):
                if hasattr(t, 'transform'):
                    t = _CallableTransformer(t.transform)
                elif callable(t):
                    t = _CallableTransformer(t)
                else:
                    raise TypeError('%r does not seem to be a transformer.' %
                                    (t,))
            return t

        if not hasattr(transformer, '__iter__'):
            dummary_t = _DummyTransformerSingleton.instance
            transformers = (
                (to_transformer(transformer),) +
                (dummary_t,) * (origin.array_count - 1)
            )
        else:
            transformers = tuple(to_transformer(t) for t in transformer)
        if len(transformers) != origin.array_count:
            raise ValueError('Number of transformers != array count.')
        self._transformers = transformers

        # inspect the data shape and type
        data_shapes = []
        data_types = []
        for inshape, intype, trans in \
                zip(self._o_data_shapes, self._o_data_types, transformers):
            shape, dtype = trans.get_shape_dtype_for(inshape, intype)
            data_shapes.append(shape)
            data_types.append(dtype)
        self._data_shapes = tuple(data_shapes)
        self._data_types = tuple(data_types)

    @property
    def data_types(self):
        return self._data_types

    @property
    def data_shapes(self):
        return self._data_shapes

    def get(self, item, array_indices=None):
        def f(origin, transformers):
            return tuple(
                t.transform(d)
                for t, d in zip(transformers, origin)
            )

        if array_indices is None:
            trans = self._transformers
        else:
            trans = tuple(self._transformers[i] for i in array_indices)

        if isinstance(item, slice):
            # If the `item` is a slice, then the output of underlying data
            # flow is ready to be transformed.
            ret = f(self._origin.get(item, array_indices=array_indices), trans)
        else:
            # If the `item` is not a slice, then it could be indices or
            # masks with any shape.  However, the transformers are supposed
            # to work only if the data is in shape of `[?] + data_shape`.
            # Thus we need to re-arrange the indices into 1-D numpy array.
            if not isinstance(item, np.ndarray):
                item = np.asarray(item)
            item_shape = item.shape
            if len(item_shape) != 1:
                item = item.reshape([-1])
                ret = tuple(
                    d.reshape(item_shape + d.shape[1:])
                    for d in f(
                        self._origin.get(item, array_indices=array_indices),
                        trans
                    )
                )
            else:
                ret = f(
                    self._origin.get(item, array_indices=array_indices),
                    trans
                )
        return ret

    def all(self):
        return tuple(
            t.transform(d)
            for t, d in zip(self._transformers, self._origin.all())
        )


class Transformer(object):
    """Base class for data transformer."""

    def get_shape_dtype_for(self, input_shape, input_dtype):
        """Get the output shape and data type for specified input.

        Parameters
        ----------
        input_shape : tuple[int]
            Shape of the input data, excluding the batch dimension.

        input_dtype : numpy.dtype
            Input data type.

        Returns
        -------
        (tuple[int], numpy.dtype)
            Output shape and data type.
        """
        raise NotImplementedError()

    def transform(self, input_data):
        """Transform the specified data.

        Parameters
        ----------
        input_data : np.ndarray
            Input data, as Numpy array.

        Returns
        -------
        np.ndarray
            Transformed data, as Numpy array.

        Notes
        -----
        A transformer should guarantee to produce identical output
        for the same inputs.
        """
        raise NotImplementedError()


class _DummyTransformer(Transformer):
    """Transformer that does no transformation."""

    def transform(self, input_data):
        return input_data

    def get_shape_dtype_for(self, input_shape, input_dtype):
        return (input_shape, input_dtype)


class _DummyTransformerSingleton:
    instance = _DummyTransformer()


class _CallableTransformer(Transformer):
    """Wrapper for callable function as transformers."""

    def __init__(self, func):
        self._func = func
        self._shape_dtype_cache = {}

    def get_shape_dtype_for(self, input_shape, input_dtype):
        cache_key = (input_shape, input_dtype)
        if cache_key not in self._shape_dtype_cache:
            self._shape_dtype_cache[cache_key] = \
                detect_output_shape_dtype(self._func, input_shape, input_dtype)
        return self._shape_dtype_cache[cache_key]

    def transform(self, input_data):
        if not isinstance(input_data, np.ndarray):
            input_data = np.asarray(input_data)
        return apply_function_on_array(self._func, input_data)
