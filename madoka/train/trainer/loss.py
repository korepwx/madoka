# -*- coding: utf-8 -*-
from datetime import datetime
from logging import getLogger

import numpy as np
import tensorflow as tf

from madoka.utils import check_argtype, round_int, INT32_MAX_VALUE
from madoka.utils.tfcompat import scalar_summary, merge_summary
from madoka.utils.tfhelper import make_function, apply_tensor_weight
from .base import Trainer
from .constraints import WeightedTrainer
from ..monitor import (SummaryMonitor, ValidationMonitor, TrainingLossMonitor,
                       CheckpointMonitor)
from ..run_steps import run_steps
from ..summary import collect_variable_summaries, SummaryWriter

__all__ = ['LossTrainer', 'WeightedLossTrainer']


class LossTrainer(Trainer):
    """Trainer that optimizes parameters by minimizing loss function."""

    def __init__(self, name='LossTrainer', **kwargs):
        super(LossTrainer, self).__init__(name=name, **kwargs)
        self._train_loss = None         # type: tf.Tensor
        self._valid_loss = None         # type: tf.Tensor
        self._loss_params = None        # type: list[tf.Variable]

    def set_loss(self, loss, placeholders=None):
        """Set the loss expression that should be minimized.

        Parameters
        ----------
        loss : tf.Tensor | (tf.Tensor, tf.Tensor)
            If one tensor is specified, it will be used as both training
            and validation loss.  Otherwise if a tuple of two tensors are
            specified, they will be used as training and validation losses
            separately.

        placeholders : tf.Tensor | collections.Iterable[tf.Tensor]
            The placeholders for loss function.

            If specified, will override the placeholders provided by
            ``set_placeholders``.
        """
        def check_loss_type(v, n):
            check_argtype(v, tf.Tensor, n)
            if v.get_shape().ndims < 1:
                import warnings
                warnings.warn(
                    '%s: Specifying scalar loss is now deprecated. '
                    'You may specify a 1-D tensor as element-wise loss.' %
                    self.name,
                    category=DeprecationWarning
                )

        # check arguments
        if isinstance(loss, tuple):
            train_loss, valid_loss = loss
        else:
            train_loss = valid_loss = loss
        check_loss_type(train_loss, 'train_loss')
        check_loss_type(valid_loss, 'valid_loss')

        if placeholders is not None:
            self.set_placeholders(placeholders)

        # memorize the loss object
        self._train_loss = train_loss
        self._valid_loss = valid_loss
        return self

    def _build_train_valid_fn(self, placeholders, train_loss, valid_loss,
                              train_flow, valid_flow):
        """Build TensorFlow functions for training and validation."""
        # get the parameters which should be optimized
        self._loss_params = params = self.trainable_vars.select()
        getLogger(__name__).debug(
            '%s: training parameters: %r.',
            self.name,
            [p.name for p in params]
        )

        # get the placeholders of the training function
        getLogger(__name__).debug(
            '%s: training inputs: %r.',
            self.name,
            [p.name for p in placeholders]
        )

        same_loss = (train_loss is valid_loss) and (
            self.apply_regularizer_to_validation or
            not self.include_global_regularizer
        )

        with self.variable_space('train_loss'):
            # compute the average of element-wise loss
            if train_loss.get_shape().ndims >= 1:
                train_loss = tf.reduce_mean(train_loss)

            # apply the global regularizer
            if self.include_global_regularizer:
                regularizer = self._books.regularization_losses
                if regularizer:
                    train_loss += tf.reduce_sum(regularizer)

        if same_loss:
            valid_loss = train_loss
        else:
            with self.variable_space('valid_loss'):
                if valid_loss.get_shape().ndims >= 1:
                    valid_loss = tf.reduce_mean(valid_loss)

        # gather summaries
        with self.variable_space('summary'):
            var_summaries = collect_variable_summaries(params)
            if var_summaries:
                summary_op = merge_summary(var_summaries)
            else:
                summary_op = None
            tloss_summary = scalar_summary('training_loss', train_loss)
            vloss_summary = scalar_summary('validation_loss', valid_loss)

        # derive the gradient updates operation
        if params:
            with self.variable_space('optimization'):
                grad_updates = self.optimizer.minimize(
                    train_loss, params=params, global_step=self.global_step)
        else:
            grad_updates = None

        # build training function
        train_fn = make_function(
            inputs=placeholders,
            outputs=[train_loss, tloss_summary],
            updates=grad_updates,
            name='train_fn'
        )

        # build validation function
        if self.validation_enabled:
            valid_fn = make_function(
                inputs=placeholders,
                outputs=[valid_loss, vloss_summary],
                name='valid_fn'
            )
        else:
            valid_fn = None

        return train_fn, valid_fn, summary_op

    def _run(self, checkpoint_dir, train_flow, valid_flow):
        if self._train_loss is None or self._valid_loss is None:
            raise RuntimeError('You should set the loss before training.')
        if self.placeholders is None:
            raise RuntimeError('You should set placeholders before training.')

        # make TensorFlow training and validation functions
        train_fn, valid_fn, summary_op = \
            self._build_train_valid_fn(self.placeholders,
                                       self._train_loss,
                                       self._valid_loss,
                                       train_flow,
                                       valid_flow)

        # prepare the training monitors here.
        monitors = list(self.monitors)

        with self.variable_space():
            # add summary monitor if required
            if self.summary_dir is None or summary_op is None:
                self._summary_writer = None
            else:
                self._summary_writer = SummaryWriter(self.summary_dir)
                monitors.append(SummaryMonitor(
                    self._summary_writer,
                    summary_op,
                    steps=self.summary_steps
                ))

            # If validation set is specified, we have to build the validation
            # function.  Otherwise we just report the training loss.
            if valid_fn is not None:
                monitors.append(ValidationMonitor(
                    valid_fn,
                    valid_flow,
                    checkpoint_dir=checkpoint_dir,
                    params=self._loss_params,
                    steps=self.validation_steps,
                    log_level=self.log_level,
                    validation_batch=self.validation_batch,
                    summary_writer=self._summary_writer
                ))
            else:
                steps = self.validation_steps or 100
                monitors.append(TrainingLossMonitor(
                    steps=steps,
                    log_level=self.log_level
                ))

            # add checkpoint monitor if required
            if self.checkpoint_enabled:
                checkpoint_monitor = CheckpointMonitor(
                    checkpoint_dir,
                    save_vars=self.restorable_vars.select(),
                    log_level=self.log_level,
                    steps=self.checkpoint_steps
                )
                monitors.append(checkpoint_monitor)

            # now it's time to run the training steps.
            if self.max_steps is None:
                max_steps = round_int(
                    self.max_epoch * self._flow.epoch_size /
                    self.batch_size
                )
            else:
                max_steps = self.max_steps
            getLogger(__name__).info(
                '%s: training started at %s.',
                self.name,
                datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            )
            run_steps(train_fn,
                      train_flow,
                      global_step=self.global_step,
                      monitors=monitors,
                      batch_size=self.batch_size,
                      max_steps=max_steps,
                      restore_max_steps=self.restore_max_steps,
                      summary_writer=self._summary_writer,
                      allow_keyboard_interrupt=self.allow_keyboard_interrupt)
            getLogger(__name__).info(
                '%s: training finished at %s.',
                self.name,
                datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            )

            return self


class WeightedLossTrainer(LossTrainer, WeightedTrainer):
    """Loss trainer with data weight."""

    def __init__(self, name='WeightedLossTrainer', **kwargs):
        super(WeightedLossTrainer, self).__init__(name=name, **kwargs)
        self._weight = None     # type: np.ndarray | tf.Variable | tf.Tensor

    def set_weight(self, weight):
        """Set the weight of data.

        Parameters
        ----------
        weight : numpy.ndarray | tf.Variable | tf.Tensor
            The weight of training data.
        """
        check_argtype(weight, (np.ndarray, tf.Variable, tf.Tensor), 'weight')
        self._weight = weight
        return self

    def set_data_flow(self, flow, valid_flow=None, shuffle=True):
        if valid_flow is not None:
            raise RuntimeError('Specifying validation flow is not supported '
                               'by WeightedLossTrainer.')
        return super(WeightedLossTrainer, self). \
            set_data_flow(flow, valid_flow, shuffle)

    def _build_weight_ph(self, dtype):
        if not isinstance(dtype, tf.DType):
            dtype = tf.as_dtype(dtype)
        return tf.placeholder(dtype, shape=(None,), name='weight')

    def _build_index_ph(self, dtype):
        if not isinstance(dtype, tf.DType):
            dtype = tf.as_dtype(dtype)
        return tf.placeholder(dtype, shape=(None,), name='index')

    def _build_index_array(self, length):
        if length <= INT32_MAX_VALUE:
            dtype = np.int32
        else:
            dtype = np.int64
        return np.arange(length, dtype=dtype)

    def _prepare_data_for_run(self, flow, valid_flow):
        assert(valid_flow is None)
        if self._weight is None:
            raise RuntimeError('Data weight has not been set.')
        if isinstance(self._weight, np.ndarray):
            if len(self._weight) != len(flow):
                raise TypeError('Length of weight array does not match '
                                'that of training flow.')
            flow = flow.merge(self._weight)
        else:
            flow = flow.merge(self._build_index_array(len(flow)))
        return super(WeightedLossTrainer, self). \
            _prepare_data_for_run(flow, valid_flow)

    def _build_train_valid_fn(self, placeholders, train_loss, valid_loss,
                              train_flow, valid_flow):
        # build the placeholder for data weights
        with self.variable_space():
            if isinstance(self._weight, np.ndarray):
                ph = self._build_weight_ph(train_loss.dtype)
                weight = ph
            else:
                ph = self._build_index_ph(train_flow.data_types[-1])
                weight = tf.gather(self._weight, ph)
            placeholders += [ph]

        # apply data weight to training and validation losses
        with self.variable_space('train_loss'):
            train_loss = apply_tensor_weight(
                train_loss, weight, name='ApplySampleWeight')
            valid_loss = apply_tensor_weight(
                valid_loss, weight, name='ApplySampleWeight')

        # call the base class to build functions.
        return super(WeightedLossTrainer, self). \
            _build_train_valid_fn(placeholders, train_loss, valid_loss,
                                  train_flow, valid_flow)
