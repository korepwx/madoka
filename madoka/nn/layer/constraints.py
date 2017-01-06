# -*- coding: utf-8 -*-
"""Layer constraint classes.

Layers may inherit from these classes, to declare some properties.
Although it is not required to do so, deriving from such classes
may help to do auxiliary check on layer instances, thus bring
benefit to debugging.
"""

class SingleOutputLayer:
    """Single output layer constraint."""


class MultiOutputLayer:
    """Multiple output layer constraint."""


class ProbaOutputLayer:
    """Probability output layer constraint.

    The ``ProbaOutputLayer`` should output a 1-dimensional non-negative
    float number vector for each input, thus can be treated as a probability
    distribution.  This constraint settle the basic interface of such layers.
    """

    @property
    def proba(self):
        """Get the output probability."""
        raise NotImplementedError()

    @property
    def label(self):
        """Get the maximum likelihood label of output probability."""
        raise NotImplementedError()

    @property
    def log_proba(self):
        """Get the log probability of output distribution."""
        raise NotImplementedError()


class SupervisedLossLayer:
    """Supervised loss layer constraint.

    A supervised loss layer should require a label to compute loss.
    Thus ``get_loss`` method will require ``target_ph`` argument.
    """

    def _validate_target(self, target_ph):
        if target_ph is None:
            raise ValueError('`target_ph` is expected to be specified '
                             'for computing supervised loss.')


class UnsupervisedLossLayer:
    """Unsupervised loss layer constraint.

    An unsupervised loss layer should not need a label to compute loss.
    Thus ``get_loss`` method will not accept ``target_ph`` argument.
    """

    def _validate_target(self, target_ph):
        if target_ph is None:
            raise ValueError('`target_ph` is expected not to be specified '
                             'for computing unsupervised loss.')
