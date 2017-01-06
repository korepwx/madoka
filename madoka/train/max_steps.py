# -*- coding: utf-8 -*-
from madoka.utils import round_int
from madoka.utils.tfhelper import (Bookkeeper, get_variable_values,
                                   VariableSetter)

__all__ = [
    'MaxSteps', 'FixedMaxSteps', 'ScalingMaxSteps', 'AddingMaxSteps'
]


class MaxSteps(object):
    """Base class for all strategies that chooses max training steps.

    It is often not easy to determine the training steps before hand,
    thus one might need to rely on some strategies to choose max steps
    as the training goes on.

    Parameters
    ----------
    initial_value : int
        Initial value for the max_steps.

    max_steps_var : tf.Variable
        Session variable for storing max_steps value.
        If not specified, will use `Bookkeeper.max_steps`.
    """

    _args = ()

    def __init__(self, initial_value, max_steps_var=None):
        self.books = Bookkeeper.for_graph()
        self.initial_value = initial_value
        if max_steps_var is None:
            max_steps_var = self.books.max_steps
            max_steps_setter = self.books.max_steps_setter
        else:
            max_steps_setter = VariableSetter(max_steps_var)
        self._var = max_steps_var
        self._setter = max_steps_setter

        # We need to delay the initialization of the max_steps value
        # until a TensorFlow session has been open.
        self._value = None  # type: int

    @property
    def value(self):
        """Get the current max_steps value."""
        return self._value

    def set_value(self, value):
        """Update both the internal and session max_steps value."""
        self._value = value
        self._setter.set(self._value)

    def init_training(self):
        """Initialize the max_steps before training.

        This method will try to recover the max_steps value from previously
        stored session, and if no value has been stored before, use the
        initial_value.
        """
        value = get_variable_values(self._var)
        if value == Bookkeeper.UNINITIALIZED_MAX_STEPS:
            value = self.initial_value
            self._setter.set(value)
        self._value = value

    def end_step(self, step, loss):
        """Update the max steps when a training step has finished.

        Parameters
        ----------
        step : int
            Global training step counter.

        loss : float
            Loss of this training step.
        """

    def end_validation(self, step, loss, best_loss):
        """Update the max steps when a validation task has finished.

        Parameters
        ----------
        step : int
            Global training step counter.

        loss : float
            Loss of this validation.

        best_loss : float
            Best validation loss in the past, excluding this time.
        """

    def __repr__(self):
        args = ('initial_value', ) + self._args
        args_repr = ','.join('%s=%s' % (k, getattr(self, k)) for k in args)
        return '%s(%s,%s)' % (self.__class__.__name__, self._value, args_repr)


class FixedMaxSteps(MaxSteps):
    """Max steps strategy that sticks to a fixed number.

    Parameters
    ----------
    value : int
        The fixed number of max_steps.

    max_steps_var : tf.Variable
        Session variable for storing max_steps value.
        If not specified, will use `Bookkeeper.max_steps`.
    """

    def __init__(self, value, max_steps_var=None):
        super(FixedMaxSteps, self).__init__(value, max_steps_var=max_steps_var)


class ScalingMaxSteps(MaxSteps):
    """Max steps increasing strategy with a scaling factor.

    This strategy increases max steps after each significant improvement
    in validation loss, as below pseudocode:

        if valid_loss < improve_threshold * best_valid_loss:
            max_steps = max(max_steps, round(scale_factor * step))

    Parameters
    ----------
    initial_value : int
        Initial max steps value.

    scale_factor : float
        Scaling factor for the max steps.

    improve_threshold : float
        Threshold for a validation loss compared to best ever loss, to be
        treated as "significant improvement".
    """

    _args = ('scale_factor', 'improve_threshold')

    def __init__(self, initial_value, scale_factor=2, improve_threshold=0.995,
                 max_steps_var=None):
        super(ScalingMaxSteps, self).__init__(
            initial_value,
            max_steps_var=max_steps_var
        )
        self.scale_factor = scale_factor
        self.improve_threshold = improve_threshold

    def end_validation(self, step, loss, best_loss):
        if loss < self.improve_threshold * best_loss:
            max_steps = max(self.value, round_int(step * self.scale_factor))
            self.set_value(max_steps)


class AddingMaxSteps(MaxSteps):
    """Max steps increasing strategy with an constant increment.

    This strategy increases max steps after each significant improvement
    in validation loss, as below pseudocode:

        if valid_loss < improve_threshold * best_valid_loss:
            max_steps = max(max_steps, step + increment)

    Parameters
    ----------
    initial_value : int
        Initial max steps value.

    increment : int
        Increment for the max steps.

    improve_threshold : float
        Threshold for a validation loss compared to best ever loss, to be
        treated as "significant improvement".
    """

    _args = ('increment', 'improve_threshold')

    def __init__(self, initial_value, increment, improve_threshold=0.995,
                 max_steps_var=None):
        super(AddingMaxSteps, self).__init__(
            initial_value,
            max_steps_var=max_steps_var
        )
        self.increment = increment
        self.improve_threshold = improve_threshold

    def end_validation(self, step, loss, best_loss):
        if loss < self.improve_threshold * best_loss:
            max_steps = max(self.value, step + self.increment)
            self.set_value(max_steps)
