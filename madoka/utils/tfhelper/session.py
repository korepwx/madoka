# -*- coding: utf-8 -*-
from collections import OrderedDict

import tensorflow as tf

from ..tfcompat import variables_initializer, global_variables

__all__ = [
    'ensure_default_session', 'get_variable_values',
    'get_variable_values_as_dict', 'get_uninitialized_variables',
    'ensure_variables_initialized',
]


def ensure_default_session():
    """Ensure that a default TensorFlow session exists, otherwise raise error.

    Returns
    -------
    tf.Session
        The default TensorFlow session.

    Raises
    ------
    RuntimeError
        If the default session does not exist.
    """
    sess = tf.get_default_session()
    if sess is None:
        raise RuntimeError('No default session has been open.')
    return sess


def get_variable_values(var_or_vars):
    """Get the values of specified TensorFlow variables.

    Parameters
    ----------
    var_or_vars : tf.Variable | collections.Iterable[tf.Variable]
        A TensorFlow variable, or a list of TensorFlow variables.

    Returns
    -------
    any | tuple[any]
        If one single variable is queried, returns its value.
        If a tuple of variables are queried, return their values in tuple.
    """
    if isinstance(var_or_vars, tf.Variable):
        return ensure_default_session().run(var_or_vars)
    else:
        var_or_vars = list(var_or_vars)
        if not var_or_vars:
            return ()
        return tuple(ensure_default_session().run(var_or_vars))


def get_variable_values_as_dict(var_or_vars):
    """Get the values of specified TensorFlow variables as dict.

    Parameters
    ----------
    var_or_vars : tf.Variable | tuple[tf.Variable]
        A TensorFlow variable, or a tuple of TensorFlow variables.

    Returns
    -------
    OrderedDict[tf.Variable, any]
        Dict from the variable instances to their fetched values.
    """
    if isinstance(var_or_vars, tf.Variable):
        var_or_vars = [var_or_vars]
    else:
        var_or_vars = list(var_or_vars)
    values = get_variable_values(var_or_vars)
    return OrderedDict((var, val) for var, val in zip(var_or_vars, values))


def get_uninitialized_variables(variables=None):
    """Get uninitialized variables as a list.

    Parameters
    ----------
    variables : collections.Iterable[tf.Variable]
        Return only uninitialized variables within this collection.
        If not specified, will return all uninitialized variables.

    Returns
    -------
    list[tf.Variable]
    """
    sess = ensure_default_session()
    if variables is None:
        variables = global_variables()
    else:
        variables = list(variables)
    init_flag = sess.run(
        tf.pack([tf.is_variable_initialized(v) for v in variables]))
    return [v for v, f in zip(variables, init_flag) if not f]


def ensure_variables_initialized(variables=None):
    """Ensure all variables are initialized.

    Parameters
    ----------
    variables : collections.Iterable[tf.Variable]
        Ensure only these variables to be initialized.
        If not specified, will ensure all variables initialized.
    """
    uninitialized = get_uninitialized_variables(variables)
    if uninitialized:
        ensure_default_session().run(variables_initializer(uninitialized))
