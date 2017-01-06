# -*- coding: utf-8 -*-
import itertools

import numpy as np
import tensorflow as tf
from sklearn.base import ClassifierMixin, RegressorMixin

from madoka.dataflow import DataFlow
from .function import make_function, Function
from .scope import ScopedObject
from ..misc import check_argtype

__all__ = [
    'batch_collect_predict',
    'BasePredictor',
    'Classifier',
    'Regressor',
]


class _PredictBatchCollector(object):
    @staticmethod
    def weighted_average(pieces):
        weight0 = len(pieces[0])
        factor = (
            weight0 /
            np.sum([len(arr) for arr in pieces]).astype(np.float64)
        )
        return np.sum([arr.sum(axis=0) * factor for arr in pieces],
                      axis=0) / weight0

    BATCH_PROCESSORS = {
        'concat': lambda v: np.concatenate(v, axis=0),
        'sum': lambda v: np.sum([a.sum(axis=0) for a in v], axis=0),
        'average': lambda v: _PredictBatchCollector.weighted_average(v)
    }
    SINGLE_BATCH_PROCESSORS = {
        'concat': lambda v: v,
        'sum': lambda v: np.sum(v, axis=0),
        'average': lambda v: np.average(v, axis=0),
    }

    def __init__(self, mode, predict_fn, batch_size=None):
        self.mode = mode
        self.predict_fn = predict_fn
        self.batch_size = batch_size

    def _collect_single(self, flow):
        ret = self.predict_fn(*flow.all())
        processor = self.SINGLE_BATCH_PROCESSORS[self.mode]
        if isinstance(ret, tuple):
            ret = tuple(processor(v) for v in ret)
        else:
            ret = processor(ret)
        return ret

    def collect(self, flow):
        # if batch_size is not set, or the data flow is empty,
        # we just collect all the prediction results in one batch.
        if self.batch_size is None or len(flow) == 0:
            return self._collect_single(flow)

        # otherwise we should build an iterator to access the data flow
        # in batches.
        it = flow.iter_epoch_batches(self.batch_size)
        result_it = (self.predict_fn(*args) for args in it)

        # consume the first batch, and detect whether or not the function
        # returns arrays in tuple
        first_result = next(result_it)
        result_in_tuple = isinstance(first_result, tuple)
        if result_in_tuple:
            array_collector = [[] for i in range(len(first_result))]
        else:
            array_collector = []

        # now, begin to consume each batch and collect in `array_collector`
        if result_in_tuple:
            for result in itertools.chain([first_result], result_it):
                for i, a in enumerate(result):
                    array_collector[i].append(a)
        else:
            for arr in itertools.chain([first_result], result_it):
                array_collector.append(arr)

        # finally, merge the batch results according to the mode
        processor = self.BATCH_PROCESSORS[self.mode]
        if result_in_tuple:
            ret = tuple(processor(a) for a in array_collector)
        else:
            ret = processor(array_collector)

        return ret


def batch_collect_predict(predict_fn, flow, batch_size=None, mode='concat'):
    """Collect prediction results in mini-batches.

    Parameters
    ----------
    predict_fn : (*args) -> np.ndarray | tuple[np.ndarray]
        The predict function, which produces array or a tuple of arrays.

    flow : DataFlow
        Input data flow.

    batch_size : int
        Mini-batch size.
        If not specified, will collect the prediction results in one batch.

    mode : str
        Way to collect the batch predicts, one of {'concat', 'sum', 'average'}.
        All of these modes will only merge the result along the first axis.

    Returns
    -------
    np.ndarray | tuple[np.ndarray]
        The merged prediction.
    """
    return _PredictBatchCollector(mode, predict_fn, batch_size).collect(flow)


class BasePredictor(ScopedObject):
    """Tensor(s) wrapper that produces some output(s).

    Parameters
    ----------
    input_ph : tf.Tensor | collections.Iterable[tf.Tensor]
        Input placeholder(s).

    output : tf.Tensor | collections.Iterable[tf.Tensor]
        Output tensor(s).

    predict_batch_size : int
        Batch size for prediction.
        If set to None, will compute prediction in one batch.

    name : str
        Name of this predictor.
    """

    def __init__(self, input_ph, output, predict_batch_size=None,
                 name='BasePredictor'):
        super(BasePredictor, self).__init__(name=name)
        if not isinstance(input_ph, tf.Tensor):
            input_ph = tuple(input_ph)
        if not isinstance(output, tf.Tensor):
            output = tuple(output)
        self.input_ph = input_ph
        self.output = output
        self.predict_batch_size = predict_batch_size

        # lazy compiled function
        self._predict_fn = None    # type: Function

    @property
    def predict_fn(self):
        """Get the TensorFlow function for output."""
        if self._predict_fn is None:
            self._predict_fn = make_function(inputs=self.input_ph,
                                             outputs=self.output,
                                             name='predict_fn')
        return self._predict_fn

    @staticmethod
    def _build_data_flow(args):
        if not args:
            raise TypeError('No input data is specified.')
        for a in args:
            if not isinstance(a, (DataFlow, np.ndarray)):
                raise TypeError(
                    '%r is neither a DataFlow nor a numpy array.' % (a,))

        if any(isinstance(a, DataFlow) for a in args):
            ret = DataFlow.empty().merge(*args)
        else:
            # if all of the arguments is numpy array, we can just build
            # a single numpy data flow.
            ret = DataFlow.from_numpy(args)
        return ret

    def _predict(self, predict_fn, *args):
        flow = self._build_data_flow(args)
        return batch_collect_predict(predict_fn, flow, self.predict_batch_size)

    def predict(self, *args):
        """Produce the output values with specified inputs.

        Parameters
        ----------
        *args : tuple[np.ndarray | DataFlow]
            Input data.

            All these arrays or data flows will be merged into a single
            data flow before feeding into the predictor function.

        Returns
        -------
        np.ndarray | tuple[np.ndarray]
            Output array or arrays.
        """
        return self._predict(self.predict_fn, *args)


class Classifier(BasePredictor, ClassifierMixin):
    """Tensor(s) wrapper that produces some probability output.

    Parameters
    ----------
    input_ph : tf.Tensor | collections.Iterable[tf.Tensor]
        Input placeholder(s).

    output : tf.Tensor
        The classification probability, as 1-D or 2-D tensor.

        If it is a 1-D tensor, or it is a 2-D tensor with shape (?, 1),
        it will be regarded as the probability of taking positive label.
        Otherwise it will be regarded as the probability of taking the
        label at each position.

    label : tf.Tensor
        If specified, will use this tensor to get the classification
        label, instead of applying `tf.argmax` on ``output``.

    log_proba : tf.Tensor
        If specified, will use this tensor to get the log-probability
        of classification, instead of applying `numpy.log` on ``output``.

    predict_batch_size : int
        Batch size for prediction.
        If set to None, will compute prediction in one batch.

    name : str
        Name of this classifier.
    """

    def __init__(self, input_ph, output, label=None, log_proba=None,
                 predict_batch_size=None, name='Classifier'):
        # check the arguments.
        check_argtype(output, tf.Tensor, 'output')
        output_shape = output.get_shape()
        if output_shape.ndims != 1 and output_shape.ndims != 2:
            raise TypeError('`output` is expected to be a 1-D or '
                            '2-D tensor.')
        if output_shape.ndims == 2 and output_shape.as_list()[1] is None:
            raise TypeError('`output` is required have a determined '
                            'shape at dimension 1.')
        if label is not None:
            check_argtype(label, tf.Tensor, 'label')

        if log_proba is not None:
            check_argtype(log_proba, tf.Tensor, 'log_proba')

        super(Classifier, self).__init__(input_ph, output, predict_batch_size,
                                         name=name)

        # memorize these arguments
        self.label = label
        self.log_proba = log_proba

        # lazy compiled functions.
        self._label_fn = None      # type: Function
        self._log_proba_fn = None  # type: Function

    @property
    def label_fn(self):
        """Get the TensorFlow function for label output."""
        if self._label_fn is None:
            label = self.label
            if label is None:
                with self.variable_space():
                    output_shape = self.output.get_shape()
                    if output_shape.ndims == 1 or \
                            output_shape.as_list()[1] == 1:
                        label = tf.cast(self.output >= 0.5, tf.int32)
                        if output_shape.ndims == 2:
                            label = tf.reshape(label, [-1])
                    else:
                        label = tf.argmax(self.output, 1)
                self.label = label
            self._label_fn = make_function(inputs=self.input_ph,
                                           outputs=label,
                                           name='label_fn')
        return self._label_fn

    @property
    def log_proba_fn(self):
        """Get the TensorFlow function for log probability output."""
        if self._log_proba_fn is None:
            if self.log_proba is None:
                with self.variable_space():
                    self.log_proba = tf.log(self.output)
            self._log_proba_fn = make_function(inputs=self.input_ph,
                                               outputs=self.log_proba,
                                               name='log_proba_fn')
        return self._log_proba_fn

    @property
    def proba(self):
        """Get the probability output tensor."""
        return self.output

    def predict(self, *args):
        """Predict the labels for given input(s).

        Parameters
        ----------
        *args : tuple[np.ndarray | DataFlow]
            Input data.

            All these arrays or data flows will be merged into a single
            data flow before feeding into the predictor function.

        Returns
        -------
        np.ndarray
            Prediction labels, as 1-D tensor.
        """
        return self._predict(self.label_fn, *args)

    def predict_proba(self, *args):
        """Predict the class probabilities for given input(s).

        Parameters
        ----------
        *args : tuple[np.ndarray | DataFlow]
            Input data.

            All these arrays or data flows will be merged into a single
            data flow before feeding into the predictor function.

        Returns
        -------
        np.ndarray
            Class probabilities.
        """
        return self._predict(self.predict_fn, *args)

    def predict_log_proba(self, *args):
        """Predict the class log-probabilities for given input(s).

        Parameters
        ----------
        *args : tuple[np.ndarray | DataFlow]
            Input data.

            All these arrays or data flows will be merged into a single
            data flow before feeding into the predictor function.

        Returns
        -------
        np.ndarray
            Class log-probabilities.
        """
        return self._predict(self.log_proba_fn, *args)


class Regressor(BasePredictor, RegressorMixin):
    """Tensor(s) wrapper that produces some value output.

    Unlike `BasePredictor`, this regressor will reshape output of size
    (?, 1) into 1-dimension.

    Parameters
    ----------
    input_ph : tf.Tensor | collections.Iterable[tf.Tensor]
        Input placeholder(s).

    output : tf.Tensor
        Output tensor.

    predict_batch_size : int
        Batch size for prediction.
        If set to None, will compute prediction in one batch.
    """

    def __init__(self, input_ph, output, predict_batch_size=None,
                 name='Regressor'):
        check_argtype(output, tf.Tensor, 'output')
        super(Regressor, self).__init__(input_ph, output, predict_batch_size,
                                        name=name)

    def predict(self, *args):
        """Predict the output value for given input(s).

        This method will reshape output of size (?, 1) into 1-dimension.

        Parameters
        ----------
        *args : tuple[np.ndarray | DataFlow]
            Input data.

            All these arrays or data flows will be merged into a single
            data flow before feeding into the predictor function.

        Returns
        -------
        np.ndarray
            Regression values.
        """
        y = self._predict(self.predict_fn, *args)
        if len(y.shape) == 2 and y.shape[1] == 1:
            y = y.reshape(y.shape[:1])
        return y
