# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from .scope import root_name_space
from ..tfcompat import global_variables

__all__ = ['GraphKeys', 'Bookkeeper', 'remove_from_collection']


class GraphKeys:
    """Additional collection keys defined by Madoka package."""
    # key to collect variables which stores states of a training session.
    TRAINING_STATES = 'training_states'

    # key to collect training phase flags.
    TRAINING_PHASE_FLAGS = 'training_phase_flags'

    # key to collect variables created by TensorFlow trainers.
    TRAINER_SLOTS = 'trainer_slots'


class Bookkeeper(object):
    """Class to manage variables via graph collections.

    This class manages TensorFlow variables via graph collections, aware of
    the other TensorFlow based libraries.

    Parameters
    ----------
    graph : tf.Graph
        TensorFlow graph instance.
        If not specified, use the current active graph.
    """

    UNINITIALIZED_MAX_STEPS = -1
    """Un-initialized max_steps value."""

    def __init__(self, graph=None):
        graph = graph or tf.get_default_graph()
        assert(graph is not None)
        self.g = graph

        # training status variables
        self.global_step = self._create_global_step()
        self.best_valid_loss = self._create_best_valid_loss()
        self.max_steps = self._create_max_steps()

        # prepare for the variable setters
        from .variables import VariableSetter
        with root_name_space():
            self.global_step_setter = VariableSetter(self.global_step)
            self.best_valid_loss_setter = VariableSetter(self.best_valid_loss)
            self.max_steps_setter = VariableSetter(self.max_steps)

        # training phase flags
        from .integration import (create_training_phase_flags,
                                  DefaultTrainingPhaseFlag)
        self._training_phase_flags = create_training_phase_flags()
        self._default_training_phase_flag = [
            flag for flag in self._training_phase_flags
            if isinstance(flag, DefaultTrainingPhaseFlag)
        ][0]

    @staticmethod
    def for_graph(graph=None):
        """Get or create the bookkeeper instance associated with graph.

        Parameters
        ----------
        graph : tf.Graph
            TensorFlow graph instance.
            If not specified, use the current active graph.

        Returns
        -------
        Bookkeeper
        """
        object_key = ('__bookkeeper',)
        graph = graph or tf.get_default_graph()
        books = graph.get_collection(object_key)
        if books:
            return books[0]
        book = Bookkeeper(graph=graph)
        graph.add_to_collection(object_key, book)
        return book

    @property
    def all_training_phase_flags(self):
        """Get all training phase flags."""
        return self._training_phase_flags

    @property
    def training_phase(self):
        """Get the default `training_phase` flag tensor."""
        return self._default_training_phase_flag

    @property
    def is_training(self):
        """Get the default `is_training` boolean flag tensor."""
        return self._default_training_phase_flag.is_training_tensor

    def _add_or_get_variable(self, name, initial_value, dtype, trainable=False,
                             **kwargs):
        """Add or get a top-level variable to the graph."""
        # Force this into the top-level name scope.
        with root_name_space(self.g):
            try:
                v = self.g.get_tensor_by_name('%s:0' % name)
            except KeyError:
                v = tf.Variable(initial_value=initial_value, name=name,
                                dtype=dtype, trainable=trainable, **kwargs)
            tf.add_to_collection(GraphKeys.TRAINING_STATES, v)
            return v

    def _create_global_step(self):
        """Create the global step variable."""
        v = self._add_or_get_variable('global_step', 0, tf.int64)
        tf.add_to_collection(tf.GraphKeys.GLOBAL_STEP, v)
        return v

    def _create_best_valid_loss(self):
        """Create the best validation loss variable."""
        return self._add_or_get_variable('best_valid_loss', np.inf, tf.float64)

    def _create_max_steps(self):
        """Create the max steps variable."""
        init_value = self.UNINITIALIZED_MAX_STEPS
        return self._add_or_get_variable('max_steps', init_value, tf.int64)

    def _create_training_phase(self):
        """Create the Madoka training phase variable."""
        v = self._add_or_get_variable('training_phase', 0, tf.int32)
        tf.add_to_collection(GraphKeys.TRAINING_PHASE_FLAGS, v)
        return v

    @property
    def global_variables(self):
        """Get all variables."""
        return global_variables()

    @property
    def trainable_variables(self):
        """Get trainable variables."""
        return tf.trainable_variables()

    @property
    def regularization_losses(self):
        """Get regularization losses defined by 3rd-party libraries."""
        return tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)


def remove_from_collection(key, item):
    """Remove the first occurrence of item from specified collection.

    Parameters
    ----------
    key : str
        Key of the collection.

    item : any
        The item to be removed from collection.

    Returns
    -------
    The `item` itself if it existed and is removed from the collection,
    or `None` if it did not exist.
    """
    c = tf.get_collection_ref(key)
    try:
        c.remove(item)
    except ValueError:
        ret = None
    else:
        ret = item
    return ret
