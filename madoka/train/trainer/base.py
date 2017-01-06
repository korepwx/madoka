# -*- coding: utf-8 -*-
import logging

import numpy as np
import tensorflow as tf

from madoka.dataflow import DataFlow
from madoka.train import Monitor
from madoka.utils import (DEFAULT_TRAIN_BATCH_SIZE, check_argtype,
                          TemporaryDirectory)
from madoka.utils import tfhelper
from madoka.utils.tfcompat import variables_initializer
from madoka.utils.tfhelper import (Bookkeeper, ScopedObject, VariableSelector,
                                   ensure_default_session)
from ..max_steps import MaxSteps
from ..optimizer import TFOptimizerWrapper, AdamOptimizer

__all__ = ['Trainer']


class Trainer(ScopedObject):
    """Base class for trainers that optimizes model parameters.

    Parameters
    ----------
    trainable_vars : VariableSelector | collections.Iterable[tf.Variable]
        Variable selector, or a list of TensorFlow variables, indicating
        the variables which should be optimized during training.

    optimizer : madoka.train.Optimizer | tf.train.Optimizer
        The TensorFlow optimizer to train model parameters.

    global_step : tf.Variable
        Variable that indicates the global training step.
        If not specified, a new global step variable will be created.

    batch_size : int
        Training batch size.

    max_epoch : int | float
        Maximum training epoch.
        Ignored if max_steps is specified.

        Since the trainer often performs training in steps rather than
        in a whole epoch, this argument can be float number.

    max_steps : int | MaxSteps
        Maximum training steps (mini-batches).

    restore_max_steps : bool
        Restore max_steps from checkpoint files.

        If set to True, the `max_steps` will be restored from checkpoint,
        in order to fully restore the last training process.

    early_stopping : bool
        Whether or not to perform early stopping by validation?
        If set to False, validation will be disabled.

    validation_split : float
        If a validation set is required to optimize the model, split this
        portion of data as the validation set.

    validation_steps : int
        Perform validation every this number of steps.
        If not specified, will automatically determine the steps.

    validation_batch : int
        Batch size for validation.
        If set to None, will compute validation loss in one batch.

    summary_dir : str
        If specified, will write training summaries to this directory.

    summary_steps : int
        Gather summaries every this number of steps.

    checkpoint_dir : str
        If specified, will save checkpoint files to this directory.

    checkpoint_steps : int
        Save checkpoints every this number of steps.

    allow_keyboard_interrupt : bool
        Whether or not to allow keyboard interrupt.

    include_global_regularizer : bool
        Whether or not to include global regularizer?

    apply_regularizer_to_validation : bool
        Whether or not to apply global regularizer to validation?

    log_level : str
        Write the training logs in this level.

    name : str
        Name of this trainer.
    """

    def __init__(self,
                 trainable_vars=VariableSelector.trainable(),
                 restorable_vars=VariableSelector.all(),
                 optimizer=AdamOptimizer(),
                 global_step=None,
                 batch_size=DEFAULT_TRAIN_BATCH_SIZE,
                 max_epoch=None,
                 max_steps=None,
                 restore_max_steps=True,
                 early_stopping=True,
                 validation_split=0.1,
                 validation_steps=None,
                 validation_batch=None,
                 summary_dir=None,
                 summary_steps=100,
                 checkpoint_dir=None,
                 checkpoint_steps=1000,
                 allow_keyboard_interrupt=True,
                 include_global_regularizer=True,
                 apply_regularizer_to_validation=True,
                 log_level=logging.INFO,
                 name='Trainer'):
        super(Trainer, self).__init__(name=name)

        # check the arguments
        if isinstance(optimizer, tf.train.Optimizer):
            optimizer = TFOptimizerWrapper(optimizer)
        if max_epoch is None and max_steps is None:
            raise TypeError('At least one of `max_epoch`, `max_steps` '
                            'should be specified.')

        # get the bookkeeper associated with current graph
        self._books = Bookkeeper.for_graph()
        if global_step is None:
            global_step = self._books.global_step

        # memorize the arguments
        self.optimizer = optimizer
        self.global_step = global_step
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.max_steps = max_steps
        self.restore_max_steps = restore_max_steps
        self.early_stopping = early_stopping
        self.validation_split = validation_split
        self.validation_steps = validation_steps
        self.validation_batch = validation_batch
        self.summary_dir = summary_dir
        self.summary_steps = summary_steps
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_steps = checkpoint_steps
        self.allow_keyboard_interrupt = allow_keyboard_interrupt
        self.include_global_regularizer = include_global_regularizer
        self.apply_regularizer_to_validation = apply_regularizer_to_validation
        self.log_level = log_level
        self.monitors = []

        # set the variable selectors
        self.trainable_vars = None          # type: VariableSelector
        self.restorable_vars = None         # type: VariableSelector
        self.set_trainable_vars(trainable_vars)
        self.set_restorable_vars(restorable_vars)

        # lazy initialized members
        self._flow = None                   # type: DataFlow
        self._valid_flow = None             # type: DataFlow
        self._placeholders = None           # type: list[tf.Tensor]
        self._label_ph = None               # type: tf.Tensor
        self._shuffle_training_flow = None  # type: bool
        self._summary_writer = None     # type: tf.train.SummaryWriter

    @property
    def validation_enabled(self):
        """Whether or not validation is enabled for this trainer?"""
        return self.early_stopping

    @property
    def checkpoint_enabled(self):
        """Whether or not checkpoint is enabled for this trainer?"""
        return self.checkpoint_dir is not None

    @property
    def summary_enabled(self):
        """Whether or not summary is enabled for this trainer?"""
        return self.summary_dir is not None

    def add_monitors(self, monitors):
        """Add training-time monitors.

        Parameters
        ----------
        monitors : Monitor | collections.Iterable[Monitor]
            Monitor or a list of monitors.

        Returns
        -------
        self
        """
        if isinstance(monitors, Monitor):
            self.monitors.append(monitors)
        else:
            self.monitors.extend(monitors)
        return self

    def clear_monitors(self):
        """Clear all monitors."""
        self.monitors.clear()
        return self

    def set_trainable_vars(self, variables):
        """Set the variables which this trainer should optimize.

        Parameters
        ----------
        variables : VariableSelector | collections.Iterable[tf.Variable]
            The variable selector, or a list of variables which should
            be optimized during training process.

        Returns
        -------
        self
        """
        if not isinstance(variables, VariableSelector):
            variables = VariableSelector().list(variables)
        self.trainable_vars = variables
        return self

    def set_restorable_vars(self, variables):
        """Set the variables to be saved in checkpoint.

        Parameters
        ----------
        variables : VariableSelector | collections.Iterable[tf.Variable]
            The variable selector, or a list of variables which should
            be optimized during training process.

        Returns
        -------
        self
        """
        if not isinstance(variables, VariableSelector):
            variables = VariableSelector().list(variables)
        self.restorable_vars = variables
        return self

    def set_placeholders(self, placeholders, label=None):
        """Set the placeholders for training and validation function.

        Parameters
        ----------
        placeholders : tf.Tensor | collections.Iterable[tf.Tensor]
            The placeholders for training and validation function.
            The order of these placeholders must match the data arrays.

        label : tf.Tensor
            Explicitly specify the label tensor.

            If not specified, the last given tensor will be considered
            as the label tensor.

        Returns
        -------
        self
        """
        if isinstance(placeholders, tf.Tensor):
            placeholders = [placeholders]
        else:
            placeholders = list(placeholders)
        if not placeholders:
            raise TypeError('No placeholder is specified.')
        for ph in placeholders:
            if not isinstance(ph, tf.Tensor):
                raise TypeError('%r is not a TensorFlow placeholder.' % (ph,))
        if label is None:
            label = placeholders[-1]
        elif label not in placeholders:
            raise ValueError('`label` is not within specified placeholders.')
        self._placeholders = placeholders
        self._label_ph = label
        return self

    @property
    def placeholders(self):
        """Get the placeholders for training and validation function.

        Returns
        -------
        list[tf.Tensor]
        """
        return self._placeholders

    @property
    def label_placeholder(self):
        """Get the label placeholder."""
        return self._label_ph

    def set_data(self, *arrays):
        """Set training data with Numpy arrays.

        If early-stopping is required, the specified data will be splitted
        into training + validation sets, where the fraction of validation set
        is determined according to ``validation_split``.

        The training data will be shuffled at each epoch, so if this
        does not satisfy your demands, you may set custom data flow objects
        by ``set_data_flow``.

        Parameters
        ----------
        *arrays : tuple[np.ndarray]
            Numpy arrays as training data.

        Returns
        -------
        self
        """
        for a in arrays:
            if not isinstance(a, np.ndarray):
                raise TypeError('%r is not a numpy array.' % (a,))
        flow = DataFlow.from_numpy(arrays)
        return self.set_data_flow(flow, shuffle=True)

    def set_data_flow(self, flow, valid_flow=None, shuffle=True):
        """Set training data with data flow.

        If validation is enabled (i.e., ``.validation_enabled == True``),
        then the specified data flow will be splitted into two partitions
        according to ``validation_split``.  The data flow will always be
        shuffled before splitting.

        Parameters
        ----------
        flow : DataFlow
            The data flow of training data.

        valid_flow : DataFlow
            DEPRECATED.

            Use this data flow as validation set, instead of splitting
            from the training data specified by `flow` argument.

            This argument has been deprecated, since some trainers might
            intend to process the data flows before splitting validation set.

        shuffle : bool
            Whether or not to shuffle the training flow at each epoch?

            A shuffled version of the given training flow will be made
            if this is set to True. (default is True)

        Returns
        -------
        self
        """
        # check the arguments
        check_argtype(flow, DataFlow, 'flow')
        if valid_flow is not None:
            import warnings
            warnings.warn('Argument `valid_flow` is deprecated.',
                          category=DeprecationWarning)
            check_argtype(valid_flow, DataFlow, 'valid_flow')

        # Here we only memorize the data flow without splitting the
        # validation set, or shuffling it, since some trainers might
        # post-process these data flows according to other arguments,
        # which can only be guaranteed to set unless `run()`.
        self._flow = flow
        self._valid_flow = valid_flow
        self._shuffle_training_flow = shuffle
        return self

    def run(self):
        """Run the trainer.

        Returns
        -------
        self
        """
        if self._flow is None:
            raise RuntimeError('You should set the data before training.')
        if self.placeholders is not None:
            if self._flow.array_count != len(self.placeholders):
                raise RuntimeError('The number of placeholders does not match '
                                   'that of data arrays.')

        train_flow, valid_flow = \
            self._prepare_data_for_run(self._flow, self._valid_flow)
        if valid_flow is not None:
            if train_flow.array_count != valid_flow.array_count:
                raise RuntimeError('The number of arrays in training flow '
                                   'does not match that of validation flow.')

        # prepare training variables
        self._prepare_variables_for_run(train_flow, valid_flow)

        # reset training states
        self._reset_training_states()

        # `checkpoint_dir` is required for early-stopping, thus if it is
        # not set, we should choose a temporary directory as the checkpoint
        # directory.
        try:
            if self.checkpoint_dir is not None:
                return self._run(self.checkpoint_dir, train_flow, valid_flow)
            else:
                with TemporaryDirectory() as tmpdir:
                    return self._run(tmpdir, train_flow, valid_flow)
        finally:
            if self._summary_writer is not None:
                self._summary_writer.close()

    def _prepare_data_for_run(self, flow, valid_flow):
        """Override this to prepare data for run."""
        if self.validation_enabled and valid_flow is None:
            train_flow, valid_flow = flow.split(
                portions=[-1, self.validation_split], shuffle=True)
        else:
            train_flow, valid_flow = flow, valid_flow
        if self._shuffle_training_flow:
            train_flow = train_flow.shuffle()
        return train_flow, valid_flow

    def _prepare_variables_for_run(self, train_flow, valid_flow):
        """Override this to prepare variables for run.

        Some trainers might need to create variables every time it runs.
        This method thus is called before `_reset_training_states()`, to
        allow the trainer preparing for such variables.
        """

    def _reset_training_states(self):
        """Override this to reset training states.

        A trainer often stores training states in TensorFlow variables.
        Such variables should be reset before each fresh training.

        Note that it is not recommended to restore from checkpoints
        in this method.  `CheckpointMonitor.before_training()` will do
        this, and `run_steps()` will call this method of each specified
        monitors.  So it's better to restore from checkpoints in `_run()`.
        """
        sess = ensure_default_session()
        reset_vars = []
        for key in (tfhelper.GraphKeys.TRAINING_STATES,):
            reset_vars.extend(tf.get_collection(key))
        sess.run(variables_initializer(reset_vars))

    def _run(self, checkpoint_dir, train_flow, valid_flow):
        raise NotImplementedError()
