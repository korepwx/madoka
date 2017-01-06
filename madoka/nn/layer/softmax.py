# -*- coding: utf-8 -*-
import tensorflow as tf

from madoka.utils import tfhelper
from madoka.utils.tfhelper import apply_tensor_weight
from .base import Layer
from .constraints import SupervisedLossLayer, ProbaOutputLayer
from .dense import DenseLayer
from ..init import Initializer, XavierNormal, Constant

__all__ = [
    'SoftmaxLayer', 'LogisticRegression',
]


class _BaseSoftmax(DenseLayer):

    def __init__(self, incoming, num_units, W=XavierNormal(), b=Constant(0.),
                 class_weight=None, name=None):
        super(_BaseSoftmax, self).__init__(
            incoming=incoming, num_units=num_units, W=W, b=b, nonlinearity=None,
            name=name
        )

        # compose the softmax output
        with self.variable_space('output'):
            self.logits = self.activation
            if class_weight is not None:
                if not isinstance(class_weight, tf.Tensor):
                    class_weight = tf.convert_to_tensor(class_weight,
                                                        dtype=self.logits.dtype)
                assert(class_weight.get_shape().ndims == 1)
                self.logits = tf.mul(self.logits, class_weight)
            self.class_weight = class_weight
            self._output = self._build_output()

    def _build_output(self):
        raise NotImplementedError()


class SoftmaxLayer(_BaseSoftmax, SupervisedLossLayer):
    """Softmax layer with loss.

    This softmax layer is not sparse, i.e., the target tensor is expected
    to have the same shape as the output tensor when computing loss.
    If a classification task is regarded, using ``LogisticRegression``
    instead of this layer.

    Parameters
    ----------
    incoming : tf.Tensor | tuple[tf.Tensor] | Layer | tuple[Layer]
        Tensor(s) or layer(s) that feed into this layer.

    num_units : int
        Number of hidden units in this softmax layer.

    W : Initializer | numpy.ndarray | tf.Variable
        Initializer, numpy array or another variable, as the weight parameter.

    b : Initializer | numpy.ndarray | tf.Variable
        Initializer, numpy array or another variable, as the bias parameter.

    class_weight : numpy.ndarray | tf.Tensor
        Weight for each class, which will be multiplied to logits.
        If not specified, all classes are supposed to have weight one.

    name : str
        Name of this layer.

    Attributes
    ----------
    num_units, nonlinearity, W, b, activation
        Attributes inherited from ``Dense`` layer.

    class_weight : numpy.ndarray | tf.Tensor
        Weight for each class.

    logits : tf.Tensor
        Linear activation of this layer, before softmax is applied.
        This is equivalent to ``activation`` attribute.

    output : tf.Tensor
        Softmax activation of this layer.
    """

    def __init__(self, incoming, num_units, W=XavierNormal(), b=Constant(0.),
                 class_weight=None, name='Softmax'):
        super(SoftmaxLayer, self).__init__(
            incoming=incoming, num_units=num_units, W=W, b=b,
            class_weight=class_weight, name=name
        )

    def get_loss(self, target_ph=None, weight=None, aggregate=False):
        self._validate_target(target_ph)
        with self.variable_space('loss'):
            labels = target_ph
            if labels.dtype != self.logits.dtype:
                labels = tf.cast(labels, self.logits.dtype)
            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits,
                labels=labels,
            )
            if weight is not None:
                loss = apply_tensor_weight(loss, weight, name='ApplyLossWeight')
            if aggregate:
                loss = tf.reduce_mean(loss)
            return loss

    def _build_output(self):
        return tf.nn.softmax(self.logits)


class LogisticRegression(_BaseSoftmax, SupervisedLossLayer, ProbaOutputLayer):
    """Sparse softmax layer with loss.

    This softmax layer is sparse, i.e., the target tensor is expected to
    be 1-dimensional integral tensor.  Each element of this tensor should
    be a one-hot encoding for original softmax regression.

    Parameters
    ----------
    incoming : tf.Tensor | tuple[tf.Tensor] | Layer | tuple[Layer]
        Tensor(s) or layer(s) that feed into this layer.

    target_num : int
        Number of classification targets of this layer.

        When specified ``target_num`` is 2 and class_weight is not specified,
        the actual output unit of this layer will be set to 1, and the layer
        activation function will be chosen as Sigmoid, so as to avoid the
        redundant parameters.  This trick is notated as `sigmoid trick`.

    W : Initializer | numpy.ndarray | tf.Variable
        Initializer, numpy array or another variable, as the weight parameter.

    b : Initializer | numpy.ndarray | tf.Variable
        Initializer, numpy array or another variable, as the bias parameter.

    class_weight : numpy.ndarray | tf.Tensor
        Weight for each class, which will be multiplied to logits.
        If not specified, all classes are supposed to have weight one.

    name : str
        Name of this layer.

    Attributes
    ----------
    num_units, nonlinearity, W, b, activation, class_weight, logits, softmax
        Attributes inherited from ``Softmax`` layer.

        If the sigmoid trick is applied (see below), then the shape of
        ``logits``, ``output`` and ``softmax`` will not match ``target_num``
        specified in the constructor.

    target_num : int
        Number of actual targets of this layer.

    sigmoid_trick : bool
        Flag to indicate whether or not the sigmoid trick has been applied.

    output : tf.Tensor
        Softmax or sigmoid activation of this layer.

        The shape of this output should match ``num_units``, but may not
        match ``target_num`` if the sigmoid trick is applied.
        You may get an pseudo-softmax output even if sigmoid trick is
        applied by using the ``proba`` attribute.
    """

    def __init__(self, incoming, target_num, W=XavierNormal(), b=Constant(0.),
                 class_weight=None, name='LogisticRegression'):
        # use sigmoid instead of softmax if num_units == 2 and class_weight
        # is not specified, so as to cut down the freedom of weights.
        self.target_num = target_num
        if class_weight is None and target_num == 2:
            target_num = 1
            self.sigmoid_trick = True
        else:
            self.sigmoid_trick = False

        super(LogisticRegression, self).__init__(
            incoming=incoming, num_units=target_num, W=W, b=b,
            class_weight=class_weight, name=name
        )

        # construct the tensor that produces label and log-proba.
        with self.variable_space('label'):
            self._label = self._build_label()

        with self.variable_space('proba'):
            self._proba = self._build_proba()

        with self.variable_space('log_proba'):
            self._log_proba = self._build_log_proba()

    def _build_output(self):
        if self.sigmoid_trick:
            return tf.nn.sigmoid(self.logits)
        else:
            return tf.nn.softmax(self.logits)

    def _build_label(self):
        if self.sigmoid_trick:
            # let sigmoid(x) = 1/(exp(-x)+1) >= 0.5, we have x >= 0.
            label = tf.reshape(tf.to_int32(self.logits >= 0), [-1])
            assert (label.dtype == tf.int32)
            return label
        else:
            return tf.cast(tf.argmax(self.logits, 1), dtype=tf.int32)

    def _build_proba(self):
        if self.sigmoid_trick:
            output = self._output
            return tf.concat(concat_dim=1, values=[1.0 - output, output])
        else:
            return self._output

    def _build_log_proba(self):
        if self.sigmoid_trick:
            # log sigmoid(x) = x - log(exp(x)+1) = x - softplus(x)
            # log(1 - sigmoid(x)) = 0 - log(exp(x)+1) = -softplus(x)
            softplus = tf.nn.softplus(self.logits)
            output = self.logits - softplus
            return tf.concat(concat_dim=1, values=[-softplus, output])
        else:
            return tf.nn.log_softmax(self.logits)

    def get_loss(self, target_ph=None, weight=None, aggregate=False):
        self._validate_target(target_ph)
        with self.variable_space('loss'):
            if self.sigmoid_trick:
                logits = tfhelper.flatten(self.logits, ndim=1)
                loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits,
                    targets=tf.cast(target_ph, logits.dtype)
                )
            else:
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits,
                    labels=target_ph
                )
            if weight is not None:
                loss = apply_tensor_weight(loss, weight, name='ApplyLossWeight')
            if aggregate:
                loss = tf.reduce_mean(loss)
            return loss

    @property
    def proba(self):
        """The softmax output of this layer.

        This shape of this attribute will always match ``target_num``
        even if sigmoid-trick is applied.

        Returns
        -------
        tf.Tensor
        """
        return self._proba

    @property
    def label(self):
        return self._label

    @property
    def log_proba(self):
        return self._log_proba
