# -*- coding: utf-8 -*-
"""Utilities for integrating with other frameworks."""

import contextlib

import tensorflow as tf

from .graph import Bookkeeper
from .scope import root_variable_space, ScopedObject
from .session import get_variable_values, ensure_default_session
from ..constant import TrainingPhase
from ..misc import unique

__all__ = [
    'TrainingPhaseFlag',
    'DefaultTrainingPhaseFlag',
    'TrainingPhaseSwitcher',
    'set_training_phase',
]


class TrainingPhaseFlag(object):
    """Abstract class for managing a training phase flag.

    Many frameworks would rely on a variable to indicate the phase of
    training, i.e., whether it is at training, validation or testing time.
    """

    def get_phase(self):
        """Get the phase code of training.

        If the phase is not set, returns `NOT_SET` phase.
        Otherwise returns one of the status code from `TrainingPhase`.

        Returns
        -------
        int
        """
        return get_variable_values(self.phase_tensor)

    def set_phase_op(self, phase):
        """Get an operation to set the phase of training.

        Parameters
        ----------
        phase : int
            Code of the phase, from `TrainingPhase`.

            If the phase is not supported, e.g., the validation or the
            testing phase, then `NOT_SET` phase should be actually set.

        Returns
        -------
        (tf.Operation, dict)
            The operation to set the phase of training, as well as the
            feed dict for this operation.
        """
        raise NotImplementedError()

    def set_phase(self, phase):
        """Set the phase of training.

        Parameters
        ----------
        phase : int
            Code of the phase, from `TrainingPhase`.

            If the phase is not supported, e.g., the validation or the
            testing phase, then `NOT_SET` phase should be actually set.
        """
        op, feed = self.set_phase_op(phase)
        ensure_default_session().run(op, feed_dict=feed)

    @property
    def phase_tensor(self):
        """Get the training phase tensor.

        Returns
        -------
        tf.Tensor
        """
        raise NotImplementedError()

    @property
    def is_training_tensor(self):
        """Get the tensor indicating whether or not it is training.

        Returns
        -------
        tf.Tensor
        """
        raise NotImplementedError()


def create_training_phase_flags():
    """Create all available training phase flags."""
    ret = [DefaultTrainingPhaseFlag()]  # type: list[TrainingPhaseFlag]
    try:
        ret.append(_TFLearnTrainingPhaseFlag())
    except ImportError:
        pass
    return ret


class _TFLearnTrainingPhaseFlag(TrainingPhaseFlag):
    """TFLearn training phase flag."""

    def __init__(self):
        import tflearn
        from .graph import GraphKeys
        with root_variable_space():
            self._mode = tflearn.get_training_mode()
            self._phase = tf.cond(
                self._mode,
                lambda: tf.constant(TrainingPhase.TRAINING, dtype=tf.int32),
                lambda: tf.constant(TrainingPhase.NOT_SET, dtype=tf.int32),
                name='training_phase'
            )
            self._phase_op = {
                True: tf.assign(self._mode, True),
                False: tf.assign(self._mode, False),
            }
        tf.add_to_collection(GraphKeys.TRAINING_PHASE_FLAGS, self._mode)

    def set_phase_op(self, phase):
        return self._phase_op[phase == TrainingPhase.TRAINING], {}

    @property
    def phase_tensor(self):
        return self._phase

    @property
    def is_training_tensor(self):
        return self._mode


class DefaultTrainingPhaseFlag(TrainingPhaseFlag):
    """The default training phase flag."""

    def __init__(self):
        from .graph import GraphKeys
        with root_variable_space():
            v = tf.get_variable('training_phase',
                                dtype=tf.int32,
                                initializer=TrainingPhase.NOT_SET,
                                trainable=False)
            tf.add_to_collection(GraphKeys.TRAINING_PHASE_FLAGS, v)
            tf.add_to_collection(GraphKeys.TRAINER_SLOTS, v)
            self._phase = v

            # we must generate the assign operations here, instead of
            # calling `set_variable_values`, otherwise the graph will
            # become modified each time `set_variable_values` is called.
            self._set_phase_op = {
                v: tf.assign(self._phase, v)
                for k, v in TrainingPhase.iter_statuses()
            }
        self._is_training = tf.equal(
            v, TrainingPhase.TRAINING, name='is_training')

    def set_phase_op(self, phase):
        try:
            return self._set_phase_op[phase], {}
        except KeyError:
            raise ValueError('%r is not a valid training phase.' % (phase,))

    @property
    def phase_tensor(self):
        return self._phase

    @property
    def is_training_tensor(self):
        return self._is_training


class TrainingPhaseSwitcher(ScopedObject):
    """Class to switch the training phase flags.

    Parameters
    ----------
    training_phase_flags : collections.Iterable[TrainingPhaseFlag]
        The training phase flags which should be switched.

        If not specified, will use all the training phase flags belonging
        to current graph.

    dataflow_context : DataFlowContext
        If specified, will also switch the training phase of this.

    name : str
        Name of this training phase switcher.
    """

    def __init__(self, training_phase_flags=None, dataflow_context=None,
                 name='TrainingPhaseSwitcher'):
        super(TrainingPhaseSwitcher, self).__init__(name=name)

        # check the arguments
        if training_phase_flags is None:
            flags = Bookkeeper.for_graph().all_training_phase_flags
        else:
            flags = list(training_phase_flags)
        self.flags = unique(flags)      # type: list[TrainingPhaseFlag]
        self.dfctx = dataflow_context

        # phase tensors of each flag
        self._phase_tensors = [f.phase_tensor for f in flags]

    @contextlib.contextmanager
    def set_phase(self, phase):
        """Temporarily switch the training phase to specified value.

        Parameters
        ----------
        phase : int
            The training phase code, from `TrainingPhase`.
        """
        sess = ensure_default_session()

        # get the original phases
        origin_phases = sess.run(self._phase_tensors)

        # gather set phase operations and restore phase operations
        set_phase_ops = []
        set_phase_feed = {}
        for f in self.flags:
            op, feed = f.set_phase_op(phase)
            set_phase_ops.append(op)
            set_phase_feed.update(feed)

        restore_phase_ops = []
        restore_phase_feed = {}
        for f, v in zip(self.flags, origin_phases):
            op, feed = f.set_phase_op(v)
            restore_phase_ops.append(op)
            restore_phase_feed.update(feed)

        # derive the partial context manager
        @contextlib.contextmanager
        def partial_set():
            try:
                sess.run(set_phase_ops, feed_dict=set_phase_feed)
                yield
            finally:
                sess.run(restore_phase_ops, feed_dict=restore_phase_feed)

        if self.dfctx:
            with self.dfctx.set_flags(phase=phase), partial_set():
                yield
        else:
            with partial_set():
                yield


@contextlib.contextmanager
def set_training_phase(phase, dataflow_context=None):
    """Set the training phase flags to specified value.

    Parameters
    ----------
    phase : int
        The training phase code, from `TrainingPhase`.

    dataflow_context : DataFlowContext
        If specified, will also switch the training phase of this.
    """
    s = TrainingPhaseSwitcher(dataflow_context=dataflow_context)
    with s.set_phase(phase):
        yield
