# -*- coding: utf-8 -*-
from logging import getLogger

import six
import tensorflow as tf

from madoka import config
from madoka.dataflow import DataFlow, DataFlowContext, get_dataflow_context
from madoka.utils import tfhelper, check_argtype, TrainingPhase
from madoka.utils.tfcompat import scalar_summary, variables_initializer
from madoka.utils.tfhelper import (Bookkeeper,
                                   get_variable_values,
                                   ensure_variables_initialized,
                                   ensure_default_session,
                                   TrainingPhaseSwitcher)
from .max_steps import MaxSteps, FixedMaxSteps
from .monitor import Monitor, MonitorChain, MonitorProxy
from .summary import SummaryWriter

__all__ = ['run_steps']


def run_steps(train_fn,
              train_flow,
              global_step=None,
              monitors=None,
              batch_size=32,
              max_steps=1000,
              restore_max_steps=True,
              summary_writer=None,
              training_loss_name='training_loss',
              allow_keyboard_interrupt=True):
    """Run training steps.

    This method will inject 3rd-party libraries, setting necessary flags for
    training, in order to make utilities like Dropout work properly.

    Parameters
    ----------
    train_fn : (*arrays) -> float | (float, tf.Summary)
        A callable function, which accepts one or more numpy arrays, and
        perform a training step.  This function should either return a
        scalar with indicates the training loss, or return a tuple which
        contains not only the training loss, but also the summary object
        for this training loss.

    train_flow : DataFlow
        The training data.

    global_step : tf.Variable
        Specify the TensorFlow variable indicating the global step.

        If specified, then the step counter will be restored from it.
        Note that this method does not update given ``global_step``
        variable.  It is a common practice for the optimizer to actually
        increase this variable.

    monitors : Monitor | collections.Iterable[Monitor]
        A monitor or a list of monitors for watching the training process.

    batch_size : int
        Size of training mini-batches.

    max_steps : int | MaxSteps
        Maximum training steps (mini-batch).

        If any one of the ``monitors`` induces early-stopping, it might take
        less than this number of steps to finish training.

    restore_max_steps : bool
        Restore max_steps from checkpoint files.

        If set to True, the `max_steps` will be restored from checkpoint,
        in order to fully restore the last training process.

    summary_writer : SummaryWriter
        If specified, will write the summary of training loss.

    training_loss_name : str
        Name for the training loss summary.

    allow_keyboard_interrupt : bool
        Whether or not to allow keyboard interrupt?

        If set to True, will suppress the KeyboardInterrupt error with only
        a warning written to the log.
        If set to False, will throw KeyboardInterrupt.
    """
    books = Bookkeeper.for_graph()
    phase_switcher = None   # type: TrainingPhaseSwitcher

    # If `train_fn` only returns the loss value, summary should be composed.
    class LossSummary:
        def __init__(self, dtype=config.floatX):
            if not isinstance(dtype, tf.DType):
                dtype = tf.as_dtype(dtype)
            self.loss_var = tf.placeholder(name='training_loss_var',
                                           shape=(),
                                           dtype=dtype)
            self.summary_op = scalar_summary(training_loss_name, self.loss_var)
    loss_summary = None

    # check the arguments.
    if monitors is None:
        monitors = Monitor()
    elif not isinstance(monitors, Monitor):
        monitors = MonitorChain(list(monitors))

    check_argtype(train_flow, DataFlow, 'train_flow')
    epoch_size = train_flow.epoch_size
    if epoch_size < batch_size:
        raise ValueError('Too few data: epoch_size < batch_size.')

    if not isinstance(max_steps, MaxSteps):
        max_steps = FixedMaxSteps(max_steps)

    # reset the training states to default values before restoring session.
    def init_training_states():
        state_vars = tf.get_collection(tfhelper.GraphKeys.TRAINING_STATES)
        init_op = variables_initializer(state_vars)
        sess = ensure_default_session()
        sess.run(init_op)
    init_training_states()

    # call ``before_training`` of the monitors to restore from saved session
    monitors.before_training()

    # if not to restore the max_steps, we need to set it to UNINITIALIZED
    if not restore_max_steps:
        # the session restorer will also restore `max_steps`, thus we
        # need to reset it to uninitialized, in order to allow us change
        # the `max_steps` by setting `trainer.max_epoch`.
        books.max_steps_setter.set(Bookkeeper.UNINITIALIZED_MAX_STEPS)

    # initialize the variables which have not been initialized.
    ensure_variables_initialized()

    # initialize the global step counter.
    if global_step is not None:
        step = int(get_variable_values(global_step))
    else:
        step = 0

    # prepare for the training.
    epoch_steps = epoch_size // batch_size
    max_steps.init_training()
    monitors.start_training(batch_size, epoch_steps, max_steps, step)

    # Utilities for setting training flag
    #
    # We keep the training flag set to True during the training process,
    # except when we're notifying monitors.
    def monitor_patch(method):
        @six.wraps(method)
        def inner(*args, **kwargs):
            with phase_switcher.set_phase(TrainingPhase.NOT_SET):
                return method(*args, **kwargs)
        return inner

    m = MonitorProxy(monitors)
    for k in ('start_epoch', 'end_epoch', 'start_step', 'end_step'):
        setattr(m, k, monitor_patch(getattr(m, k)))

    try:
        # initialize the data flow context, or get an existing one.
        ctx = get_dataflow_context()
        if ctx is None or train_flow not in ctx.flows:
            ctx = DataFlowContext([train_flow])
        phase_switcher = TrainingPhaseSwitcher(dataflow_context=ctx)

        # the out loop indicates the pass of data (or to say, the epochs)
        epoch = 0
        with ctx.as_default(), phase_switcher.set_phase(TrainingPhase.TRAINING):
            while step < max_steps.value:
                # Checking `step < max_steps.value` is necessary.
                #
                # When recovering from a previous training which has already
                # done the main training process but failed at `end_training`,
                # the step will equal to `max_steps.value`.
                m.start_epoch(epoch)
                n_batches = 0
                total_loss = 0

                # the inner loop indicates the mini-batches of data.
                for arrays in train_flow.iter_epoch_batches(
                        batch_size, ignore_incomplete_batch=True):
                    m.start_step(step)

                    # set `is_training` to True when calling training function.
                    result = train_fn(*arrays)

                    if isinstance(result, (tuple, list)):
                        loss, summary = result[0], result[1]
                    else:
                        if loss_summary is None:
                            loss_summary = LossSummary()
                        loss = result
                        summary = loss_summary.summary_op
                    m.end_step(step, loss)
                    max_steps.end_step(step, loss)

                    # try to add the summary of training loss
                    if summary is not None and summary_writer is not None:
                        if loss_summary and summary is loss_summary.summary_op:
                            summary_writer.write(
                                summary,
                                global_step=step,
                                givens={loss_summary.loss_var: loss}
                            )
                        else:
                            summary_writer.write(summary, global_step=step)

                    n_batches += 1
                    total_loss += loss
                    step += 1

                    if step >= max_steps.value or m.is_inducing_stopping:
                        break

                m.end_epoch(epoch, float(total_loss) / n_batches)
                epoch += 1

                if step >= max_steps.value or m.is_inducing_stopping:
                    break

                # prepare data flow for the next epoch
                ctx.reset_epoch()

        # complete the training.
        monitors.end_training(has_error=False)
    except KeyboardInterrupt:
        if not allow_keyboard_interrupt:
            monitors.end_training(has_error=True)
            raise
        else:
            monitors.end_training(has_error=False)
            getLogger(__name__).warn('Keyboard interrupted.')
    except:
        monitors.end_training(has_error=True)
        raise
