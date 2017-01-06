# -*- coding: utf-8 -*-
import tensorflow as tf

from ..session import ensure_default_session

__all__ = ['VariableSetter']


class VariableSetter(object):
    """Object for setting values of some TensorFlow variables.

    It is very common to assign values to some variables, however, calling
    `tf.assign` will create a new operation in the graph.  This will quickly
    become a mass of trouble if repeated doing so in a training loop.

    Thus we need to store the assignment operation, using a placeholder
    to feed values into the variables.

    Parameters
    ----------
    variables : tf.Variable | collections.Iterable[tf.Variable]
        Variables whose values would be set.

    name : str
        Name of this variable setter.
    """

    def __init__(self, variables, name='VariableSetter'):
        if isinstance(variables, tf.Variable):
            variables = [variables]
            self.single_var = True
        else:
            variables = list(variables)
            self.single_var = False
        self.variables = variables      # type: list[tf.Variable]
        with tf.name_scope(name):
            self.placeholders = [
                tf.placeholder(v.dtype, v.get_shape().as_list(),
                               '%s_input' % v.name.split(':')[0])
                for v in self.variables
            ]
            self.assign_op = tf.group(*(
                tf.assign(var, ph)
                for var, ph in zip(self.variables, self.placeholders)
            ))

    def set(self, values):
        """Set values to the variables.

        Parameters
        ----------
        values : any | collections.Iterable[any]
            If a single variable is specified in the constructor,
            this argument should be directly specified the value.

            Otherwise a list of values should be specified, matching
            the order of variables specified in the constructor.
        """
        if self.single_var:
            values = [values]
        else:
            values = list(values)
        if len(values) != len(self.variables):
            raise TypeError(
                'Expected %s values but got %s.' %
                (len(self.variables), len(values))
            )
        ensure_default_session().run(self.assign_op, feed_dict={
            ph: val for ph, val in zip(self.placeholders, values)
        })
