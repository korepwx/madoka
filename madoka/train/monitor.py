# -*- coding: utf-8 -*-
import logging
import math
import os
import time
from datetime import datetime
from logging import getLogger

import numpy as np
import tensorflow as tf

from madoka import config
from madoka.dataflow import DataFlow
from madoka.utils.tfcompat import scalar_summary
from madoka.utils.tfhelper import (Bookkeeper, ScopedObject, SessionRestorer,
                                   ensure_default_session, get_variable_values)
from .max_steps import MaxSteps

__all__ = [
    'Monitor',
    'MonitorProxy',
    'MonitorChain',
    'ValidationMonitor',
    'PeriodicalMonitor',
    'CheckpointMonitor',
    'SummaryMonitor',
    'TrainingLossMonitor'
]


class Monitor(ScopedObject):
    """Base class that monitors training process."""

    def before_training(self):
        """Notify the monitor even before ``start_training``.

        Some monitors may need this stage to restore from saved session.
        """

    def start_training(self, batch_size, epoch_steps, max_steps, initial_step):
        """Notify the monitor that a training process will start.

        Parameters
        ----------
        batch_size : int
            Size of each training step (mini-batch).

        epoch_steps : int
            Estimated number of steps in each epoch.
            The steps in each epoch is not necessarily same.

        max_steps : MaxSteps
            Hard limit of total steps.

        initial_step : int
            The initial step recovered from checkpoint.
        """

    def end_training(self, has_error=False):
        """Notify the monitor that a training process has finished.

        It will be triggered whether or not any error has taken place.

        Parameters
        ----------
        has_error : bool
            Whether or not any error has occurred during training.
        """

    def start_epoch(self, epoch):
        """Notify the monitor that a training epoch will start.

        Parameters
        ----------
        epoch : int
            Index of the epoch, starting from 0.
        """

    def end_epoch(self, epoch, avg_loss):
        """Notify the monitor that a training epoch has completed.

        Parameters
        ----------
        epoch : int
            Index of the epoch, starting from 0.

        avg_loss : float
            Average training loss of all steps in this epoch.
            Would be None if the training process does not evolve a loss.
        """

    def start_step(self, step):
        """Notify the monitor that a training step (mini-batch) will start.

        Parameters
        ----------
        step : int
            Index of the step, starting from 0.

            This should be the total number of steps have ever been performed
            since the whole training process started, not from the start of
            this epoch.
        """

    def end_step(self, step, loss):
        """Notify the monitor that a training step (mini-batch) has completed.

        Parameters
        ----------
        step : int
            Index of the step, starting from 0.

        loss : float
            Training loss of this step.
            Would be None if the training process does not evolve a loss.
        """

    @property
    def is_inducing_stopping(self):
        """Whether or not this monitor is inducing early-stopping?

        Monitors have no way to stop a training process immediately unless
        by raising an error. This flag can only recommend the outside training
        procedure to exit.

        Returns
        -------
        bool
        """
        return False


class MonitorProxy(Monitor):
    """Proxy of another monitor.

    Parameters
    ----------
    monitor : Monitor
        The underlying monitor.

    name : str
        Name of this monitor.
    """

    def __init__(self, monitor, name=None):
        super(MonitorProxy, self).__init__(name=name)
        self.m = monitor

    def before_training(self):
        self.m.before_training()

    def start_training(self, batch_size, epoch_steps, max_steps, initial_step):
        self.m.start_training(batch_size, epoch_steps, max_steps, initial_step)

    def end_training(self, has_error=False):
        self.m.end_training(has_error=has_error)

    def start_epoch(self, epoch):
        self.m.start_epoch(epoch)

    def end_epoch(self, epoch, avg_loss):
        self.m.end_epoch(epoch, avg_loss)

    def start_step(self, step):
        self.m.start_step(step)

    def end_step(self, step, loss):
        self.m.end_step(step, loss)

    @property
    def is_inducing_stopping(self):
        return self.m.is_inducing_stopping


class MonitorChain(Monitor):
    """Chain of monitors, aggregating multiple monitors into one.

    Methods of the monitors in this chain would be called one by one, in
    determined order.  If any one of the monitors is inducing early-stopping,
    then the whole chain would do so.

    Parameters
    ----------
    monitor_or_monitors : Monitor | collections.Iterable[Monitor]
        A monitor, or a list of monitors.

    name : str
        Name of this monitor.
    """

    def __init__(self, monitor_or_monitors, name=None):
        super(MonitorChain, self).__init__(name=name)
        if isinstance(monitor_or_monitors, Monitor):
            self.monitors = [monitor_or_monitors]
        else:
            self.monitors = list(monitor_or_monitors)

    def before_training(self):
        for m in self.monitors:
            m.before_training()

    def start_training(self, batch_size, epoch_steps, max_steps, initial_step):
        for m in self.monitors:
            m.start_training(batch_size, epoch_steps, max_steps, initial_step)

    def end_training(self, has_error=False):
        for m in self.monitors:
            m.end_training(has_error=has_error)

    def start_epoch(self, epoch):
        for m in self.monitors:
            m.start_epoch(epoch)

    def end_epoch(self, epoch, avg_loss):
        for m in self.monitors:
            m.end_epoch(epoch, avg_loss)

    def start_step(self, step):
        for m in self.monitors:
            m.start_step(step)

    def end_step(self, step, loss):
        for m in self.monitors:
            m.end_step(step, loss)

    @property
    def is_inducing_stopping(self):
        return any(m.is_inducing_stopping for m in self.monitors)


class ValidationMonitor(Monitor):
    """Monitor that performs validation and early-stopping.

    This monitor computes the loss on validation set every few steps,
    and use the validation loss to determine whether or not to accept
    the current set of parameters.

    Parameters
    ----------
    valid_fn : () -> float | tf.Summary
        Callable function to perform a validation pass.
        This function should either return a scalar which indicates the
        training loss, or return a tuple which contains not only the training
        loss, but also the summary object for the loss.

    valid_data : DataFlow
        A data flow object as the validation data.

    checkpoint_dir : str
        Required for storing the parameters of best validation loss.
        These parameters will be saved to `checkpoint_dir + "/best_valid.ckpt"`.

    params : collections.iterable[tf.Variable]
        List of parameters that should be regularized by early-stopping.

    steps : int
        Perform validation every this number of steps.

        If not specified, will automatically choose one according to the
        data count of validation set and the training batch size.

    validation_batch : int
        Validation batch size.  If not specified, will compute validation loss
        in one batch.

    validation_loss_name : str
        Alternative name of validation loss in summary.

    log_level : str
        Log validation loss in this level.

    summary_writer : SummaryWriter
        If specified, will output the summary of validation loss.

    name : str
        Name of this monitor.
    """

    def __init__(self,
                 valid_fn,
                 valid_data,
                 checkpoint_dir,
                 params,
                 steps=None,
                 validation_batch=None,
                 validation_loss_name=None,
                 log_level=logging.INFO,
                 summary_writer=None,
                 name='ValidationMonitor'):
        if not isinstance(valid_data, DataFlow):
            raise TypeError('`valid_data` is expected to be DataFlow, '
                            'but got %r.' % (valid_data,))
        super(ValidationMonitor, self).__init__(name=name)
        self._valid_fn = valid_fn
        self._valid_data = valid_data
        self._params = list(params)
        self._steps = steps
        self._valid_batch = validation_batch
        self._valid_loss_name = validation_loss_name
        if os.path.exists(checkpoint_dir):
            if not os.path.isdir(checkpoint_dir):
                raise IOError('%r exists but is not a directory.')
        else:
            os.makedirs(checkpoint_dir)
        self._checkpoint_dir = checkpoint_dir
        self._log_level = log_level
        self._summary_writer = summary_writer

        # get the bookkeeper associated with current graph
        self._books = Bookkeeper.for_graph()

        # the restorer for early-stopping
        self._restorer = None  # type: SessionRestorer

        # loss variable and loss summary operation.
        self._loss_ph = None
        self._summary_op = None

        # sum of the training loss since last report
        self._train_loss_sum = None
        # number of training loss since last report
        self._train_loss_num = None

        # this monitor will do validation every this number of steps
        # (guaranteed not None after training started).
        self._actual_steps = None

        # memorize the given max_steps instance
        self._max_steps = None  # type: MaxSteps

        # list of parameters that should be monitored
        self._monitored_params = None
        # best validation loss
        self._best_valid_loss = None
        # whether or not we've just did validation last step?
        self._last_step_validated = None

    @property
    def _checkpoint_file(self):
        return os.path.join(self._checkpoint_dir, 'best_valid.ckpt')

    @property
    def _checkpoint_manifest(self):
        return 'best_valid.latest'

    def start_training(self, batch_size, epoch_steps, max_steps, initial_step):
        sess = ensure_default_session()

        # in case the validation function does not return summary, or we
        # perform validation in mini-batches, we would have to construct the
        # loss summary manually.
        with self.variable_space():
            valid_loss_name = self._valid_loss_name or 'validation_loss'
            self._loss_ph = tf.placeholder(name='validation_loss_ph',
                                           shape=(),
                                           dtype=tf.as_dtype(config.floatX))
            self._summary_op = scalar_summary(valid_loss_name, self._loss_ph)

        # clear the training loss sum
        self._train_loss_sum = self._train_loss_num = 0

        # memorize the max_steps instance
        self._max_steps = max_steps

        # determine the step interval.
        if self._steps is None:
            num_examples = self._valid_data.epoch_size
            # Automatically determine the step interval, such that:
            #
            # 1. At least the same number of training data is used before
            #    using the validation data.
            # 2. Validation step should no less than min(100, max_steps * 0.1)
            # 3. A multiple of 10, 100 or 1000, etc, according to the
            #    step-interval selected from previous rule.
            max_steps_v = max_steps.value
            actual_steps = (num_examples + batch_size - 1) // batch_size
            actual_steps = max(min(100, int(max_steps_v * 0.1)), actual_steps)
            ten_base = 10 ** int(math.log(actual_steps, 10))
            self._actual_steps = \
                ((actual_steps + ten_base - 1) // ten_base) * ten_base
        else:
            self._actual_steps = self._steps

        # prepare for parameters for early-stopping.
        self._monitored_params = self._params

        # create the saver for early-stopping
        best_valid_loss_var = self._books.best_valid_loss
        with self.variable_space():
            vars = self._monitored_params + [best_valid_loss_var]
            self._restorer = SessionRestorer(
                vars,
                self._checkpoint_file,
                latest_file=self._checkpoint_manifest,
                save_meta=False,
                name='ValidSaver'
            )

        # restore best parameters from the saver.
        chkfile = self._restorer.get_latest_file()
        if chkfile:
            # Here we should only restore `best_valid_loss`!
            # The active parameters are managed by CheckpointMonitor, instead
            # of by this ValidationMonitor.
            loss_saver = tf.train.Saver(var_list=[best_valid_loss_var],
                                        name='BestValidLossLoader')
            loss_saver.restore(sess, chkfile)
            self._best_valid_loss = get_variable_values(best_valid_loss_var)
        else:
            self._best_valid_loss = np.inf

        # set the start time stamp
        if self._log_level:
            if initial_step > 0:
                getLogger(__name__).log(
                    self._log_level,
                    'Validation recovered at step %d; '
                    'best validation loss %.6f.',
                    initial_step, self._best_valid_loss
                )
            else:
                getLogger(__name__).log(self._log_level, 'Validation enabled.')

        # set the flag that we've never done validation
        self._last_step_validated = False

    def _do_validation(self, step, train_loss):
        start_valid_time = time.time()

        # compute the validation loss.
        valid_weights = []
        valid_result = []

        if self._valid_batch is not None:
            batch_size = self._valid_batch
            for args in self._valid_data.iter_epoch_batches(batch_size):
                valid_weights.append(len(args[0]))
                valid_result.append(self._valid_fn(*args))
        else:
            args = self._valid_data.all()
            valid_weights.append(len(args[0]))
            valid_result.append(self._valid_fn(*args))

        if len(valid_result) == 0:
            raise RuntimeError('No validation data.')
        elif len(valid_result) == 1:
            if isinstance(valid_result[0], (tuple, list)):
                loss, summary = valid_result[0]
            else:
                loss = valid_result[0]
                summary = self._summary_op
        else:
            # we've performed validation in mini-batches, thus we must compose
            # the summary ourselves.
            weights = (
                np.array(valid_weights) /
                np.sum(valid_weights).astype(np.float32)
            )
            losses = np.array(
                [v[0] if isinstance(v, (tuple, list)) else v
                 for v in valid_result]
            )
            loss = np.sum(weights * losses)
            summary = self._summary_op

        if self._summary_writer is not None and \
                summary is not None and step is not None:
            self._summary_writer.write(summary,
                                       global_step=step,
                                       givens={self._loss_ph: loss})

        # Notify the max_steps instance that we've done a validation task.
        # We only do notification if it's really at end of a step, not end
        # of the training (according to `step is not None`)
        if step is not None:
            self._max_steps.end_validation(step, loss, self._best_valid_loss)

        best_params_updated = False
        if loss < self._best_valid_loss:
            # now save the best parameters to checkpoint file
            self._books.best_valid_loss_setter.set(loss)
            # we use versioned save with global_step here, in order to
            # ensure the checkpoint is written successfully.
            self._restorer.save(global_step=step)
            self._best_valid_loss = loss
            best_params_updated = True

        # report the loss if required
        if step is not None and self._log_level:
            best_mark = ' (*)' if best_params_updated else ''
            valid_time_usage = time.time() - start_valid_time
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            getLogger(__name__).log(
                self._log_level,
                'Step %d/%d: at %s, average train loss %.6f, valid loss %.6f; '
                'validated in %.2f secs.%s',
                step, self._max_steps.value, time_str, train_loss, loss,
                valid_time_usage, best_mark
            )

    def end_step(self, step, loss):
        # sum up training loss
        self._train_loss_sum += loss
        self._train_loss_num += 1

        # Do validation if necessary.
        #
        # Note that `step` counts from zero, but we want to make validation
        # not at `*99` steps. so we choose to skip the first step.
        if step > 0 and step % self._actual_steps == 0:
            train_loss = self._train_loss_sum / float(self._train_loss_num)
            self._do_validation(step, train_loss)
            self._train_loss_sum = self._train_loss_num = 0
            self._last_step_validated = True
        else:
            self._last_step_validated = False

    def end_training(self, has_error=False):
        if not has_error:
            # perform the final validation if there's some more training since
            # the last validation.
            if not self._last_step_validated:
                self._do_validation(None, None)
            # restore the best ever params.
            self._restorer.restore()
        # NOTE: DO NOT DELETE CHECKPOINT FILE HERE!
        #
        # Let the caller to delete the whole ``checkpoint_dir`` until
        # it has already saved the model to somewhere else.
        #
        # Finally, we should clear the recorded best params in the session.
        self._monitored_params = None
        self._best_valid_loss = None
        self._last_step_validated = None


class PeriodicalMonitor(Monitor):
    """Monitor that runs every few steps or seconds.

    Parameters
    ----------
    seconds : int
        Run this monitor every this number of seconds.

    steps : int
        Run this monitor every this number of steps.

    name : str
        Name of this monitor.
    """

    def __init__(self, seconds=None, steps=None, name='PeriodicalMonitor'):
        super(PeriodicalMonitor, self).__init__(name=name)
        if seconds is None and steps is None:
            raise ValueError(
                'At least either "seconds" or "steps" should be specified.')
        self._seconds = seconds
        self._steps = steps

        # last checkpoint time and step
        self._last_chk_time = None
        self._last_chk_step = None

    def start_training(self, batch_size, epoch_steps, max_steps, initial_step):
        self._last_chk_time = time.time()
        if self._steps is not None:
            self._last_chk_step = int(initial_step // self._steps) * self._steps
        else:
            self._last_chk_step = 0

    def _run(self, step, loss, now_time):
        """Derived classes should override this to actually run monitor."""
        raise NotImplementedError()

    def _should_run(self, step):
        if self._steps is not None:
            if (step - self._last_chk_step) >= self._steps:
                return True
        if self._seconds is not None:
            if (time.time() - self._last_chk_time) >= self._seconds:
                return True
        return False

    def end_step(self, step, loss):
        if self._should_run(step):
            now_time = time.time()
            self._run(step, loss, now_time)
            self._last_chk_time = time.time()
            self._last_chk_step = step


class CheckpointMonitor(PeriodicalMonitor):
    """Monitor that saves checkpoint every few steps or seconds.

    Parameters
    ----------
    checkpoint_dir : str
        Directory where the checkpoint file should be stored.
        The checkpoint will be saved to `checkpoint_dir + "/train.ckpt"`.

    save_vars : collections.Iterable[tf.Variable]
        List of variables that should be saved. (required)

    log_level : str
        Log the message of saving checkpoint in this level.

    seconds : int
        Run this monitor every this number of seconds.

    steps : int
        Run this monitor every this number of steps.

    name : str
        Name of this monitor.
    """

    def __init__(self, checkpoint_dir, save_vars, log_level=logging.INFO,
                 seconds=None, steps=None, name='CheckpointMonitor'):
        if os.path.exists(checkpoint_dir):
            if not os.path.isdir(checkpoint_dir):
                raise IOError('%r exists but is not a directory.')
        else:
            os.makedirs(checkpoint_dir)
        super(CheckpointMonitor, self).__init__(seconds=seconds,
                                                steps=steps,
                                                name=name)

        # get the bookkeeper associated with current graph
        self._books = Bookkeeper.for_graph()

        # memorize arguments
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        self._save_vars = list(save_vars)
        self._log_level = log_level
        self._graph = tf.get_default_graph()
        self._restorer = None   # type: SessionRestorer

        getLogger(__name__).debug('Restorable variables: %s',
                                  [v.name for v in self._save_vars])

    @property
    def _checkpoint_file(self):
        return os.path.join(self.checkpoint_dir, 'train.ckpt')

    @property
    def _checkpoint_manifest(self):
        return 'train.latest'

    def before_training(self):
        # restore the session at `before_training` stage.
        self._restorer = SessionRestorer(
            self._save_vars, self._checkpoint_file,
            latest_file=self._checkpoint_manifest
        )
        self._restorer.restore()

    def _run(self, step, loss, now_time):
        self._restorer.save(step)
        if self._log_level:
            time_str = datetime.strftime(datetime.fromtimestamp(now_time),
                                         '%Y-%m-%d %H:%M:%S')
            getLogger(__name__).log(
                self._log_level,
                'Checkpoint saved at step %d, %s.',
                step, time_str
            )


class SummaryMonitor(PeriodicalMonitor):
    """Monitor that writes summary every few steps or seconds.

    Parameters
    ----------
    writer : madoka.train.SummaryWriter
        Wrapped TensorFlow summary writer.

    summary : tf.Tensor | tf.Variable | tf.Summary | list[tf.Summary]
        Any object that is writable by the summary writer.

    seconds : int
        Run this monitor every this number of seconds.

    steps : int
        Run this monitor every this number of steps.

    name : str
        Name of this monitor.
    """

    def __init__(self, writer, summary, seconds=None, steps=None,
                 name='SummaryMonitor'):
        super(SummaryMonitor, self).__init__(seconds, steps, name=name)
        self._writer = writer
        self._summary = summary

    def _run(self, step, loss, now_time):
        self._writer.write(self._summary, step)


class TrainingLossMonitor(PeriodicalMonitor):
    """Monitor that prints the average training loss every few steps or seconds.

    Parameters
    ----------
    seconds : int
        Run this monitor every this number of seconds.

    steps : int
        Run this monitor every this number of steps.

    log_level : str
        Log the training loss in this level.

    name : str
        Name of this monitor.
    """

    def __init__(self, seconds=None, steps=None, log_level=logging.INFO,
                 name='TrainingLossMonitor'):
        super(TrainingLossMonitor, self).__init__(seconds, steps, name=name)
        self._sum_loss = self._num_steps = None
        self._log_level = log_level
        self._max_steps = None  # type: MaxSteps

    def start_training(self, batch_size, epoch_steps, max_steps, initial_step):
        self._max_steps = max_steps
        self._sum_loss = self._num_steps = 0
        super(TrainingLossMonitor, self).start_training(batch_size,
                                                        epoch_steps,
                                                        max_steps,
                                                        initial_step)

    def end_step(self, step, loss):
        self._sum_loss += loss
        self._num_steps += 1
        super(TrainingLossMonitor, self).end_step(step, loss)

    @property
    def avg_loss(self):
        return self._sum_loss / float(self._num_steps)

    def _run(self, step, loss, now_time):
        if self._num_steps > 0 and self._log_level:
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            getLogger(__name__).log(
                self._log_level,
                'Step %d/%d: at %s, average train loss %.6f.',
                step, self._max_steps.value, time_str, self.avg_loss
            )
        self._num_steps = self._sum_loss = 0
