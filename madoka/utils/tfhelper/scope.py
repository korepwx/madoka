# -*- coding: utf-8 -*-

"""Enhanced functions to open TensorFlow name and variable scopes.

In order to be different from TensorFlow functions, we use "space" instead
of "scope" in our function names.
"""

import contextlib

import six
import tensorflow as tf
from tensorflow.python.ops.variable_scope import (_VARSCOPE_KEY,
                                                  _get_default_variable_store)

from madoka import config

__all__ = [
    'root_name_space',
    'get_variable_space',
    'root_variable_space',
    'variable_space',
    'ScopedObject',
]


@contextlib.contextmanager
def root_name_space(graph=None):
    """Select the root name scope of specified graph.

    Parameters
    ----------
    graph : tf.Graph
        The graph whose root scope should be selected.
        If not specified, will choose the default graph.
    """
    if graph is None:
        graph = tf.get_default_graph()
    with graph.as_default(), graph.name_scope(None) as scope:
        yield scope


@contextlib.contextmanager
def _pure_variable_scope(name_or_scope,     # full name of the scope
                         reuse=None,
                         initializer=None,
                         regularizer=None,
                         caching_device=None,
                         partitioner=None,
                         custom_getter=None,
                         dtype=config.floatX):
    # MODIFIED FROM TENSORFLOW ORIGINAL SOURCE CODE (r0.11)

    if not isinstance(dtype, tf.DType):
        dtype = tf.as_dtype(dtype)
    # Get the reference to the collection as we want to modify it in place.
    default_varscope = tf.get_collection_ref(_VARSCOPE_KEY)
    if default_varscope:
        old = default_varscope[0]
    else:
        default_varscope[:] = [None]
        old = None
    var_store = _get_default_variable_store()
    if isinstance(name_or_scope, tf.VariableScope):
        scope_name = name_or_scope.name or ''
    else:
        scope_name = name_or_scope.rstrip('/')
    try:
        if scope_name:
            var_store.open_variable_scope(scope_name)
        if isinstance(name_or_scope, tf.VariableScope):
            default_varscope[0] = tf.VariableScope(
                name_or_scope.reuse if reuse is None else reuse,
                name=scope_name,
                initializer=name_or_scope.initializer,
                regularizer=name_or_scope.regularizer,
                caching_device=name_or_scope.caching_device,
                partitioner=name_or_scope.partitioner,
                dtype=name_or_scope.dtype,
                custom_getter=name_or_scope.custom_getter,
                name_scope=scope_name + '/' if scope_name else '')
        else:
            default_varscope[0] = tf.VariableScope(
                not not (old and old.reuse) if reuse is None else reuse,
                name=scope_name,
                initializer=old and old.initializer,
                regularizer=old and old.regularizer,
                caching_device=old and old.caching_device,
                partitioner=old and old.partitioner,
                dtype=(old and old.dtype) or dtype,
                custom_getter=old and old.custom_getter,
                name_scope=scope_name + '/' if scope_name else '')
        if initializer is not None:
            default_varscope[0].set_initializer(initializer)
        if regularizer is not None:
            default_varscope[0].set_regularizer(regularizer)
        if caching_device is not None:
            default_varscope[0].set_caching_device(caching_device)
        if partitioner is not None:
            default_varscope[0].set_partitioner(partitioner)
        if custom_getter is not None:
            default_varscope[0].set_custom_getter(custom_getter)
        if dtype is not None:
            default_varscope[0].set_dtype(dtype)
        yield default_varscope[0]
    finally:
        if scope_name:
            var_store.close_variable_subscopes(scope_name)
        if old is None:
            default_varscope[:] = []
        else:
            default_varscope[0] = old


def get_variable_space():
    """Get the current variable space.

    Returns
    -------
    tf.VariableScope
    """
    return tf.get_variable_scope()


@contextlib.contextmanager
def root_variable_space(reuse=None,
                        initializer=None,
                        regularizer=None,
                        caching_device=None,
                        partitioner=None,
                        custom_getter=None,
                        dtype=config.floatX):
    """Enter root variable scope of current graph.

    The settings of current opened variable scope will be copied to the
    returned root variable scoped, unless specified in arguments.

    Parameters
    ----------
    reuse : bool | None
        If `True` or `False`, set the reuse mode of returned variable scope.
        If `None`, inherit the current scope reuse.

    initializer, regularizer, caching_device, partitioner, custom_getter, dtype
        See tf.variable_scope for more details.

    Yields
    ------
    tf.VariableScope
        The root variable scope.
    """
    with root_name_space():
        with _pure_variable_scope(
                '',
                reuse=reuse,
                initializer=initializer,
                regularizer=regularizer,
                caching_device=caching_device,
                partitioner=partitioner,
                custom_getter=custom_getter,
                dtype=dtype) as scope:
            yield scope


@contextlib.contextmanager
def variable_space(name_or_scope,
                   default_name=None,
                   reuse=None,
                   initializer=None,
                   regularizer=None,
                   caching_device=None,
                   partitioner=None,
                   custom_getter=None,
                   dtype=config.floatX):
    """Open a variable scope for defining ops and creating variables.

    Unlike `tf.variable_scope`, which might open existing variable scopes
    with name scopes with different names (e.g., re-open "var_scope1"
    might actually open "var_scope1_1" name scope), this function guarantees
    to open exactly the same name scope as the name of the variable scope.

    Parameters
    ----------
    name_or_scope : str | tf.VariableScope | None
        The scope to open.

    default_name : str
        The default name to use if the `name_or_scope` argument is `None`,
        which will be uniquified.  If `name_or_scope` is provided it won't
        be used and therefore it is not required and can be None.

    reuse : bool | None
        If `True` or `False`, set the reuse mode of returned variable scope.
        If `None`, inherit the current scope reuse.

    initializer, regularizer, caching_device, partitioner, custom_getter, dtype
        See tf.variable_scope for more details.
    """
    if name_or_scope is None and default_name is None:
        raise TypeError('At least one of `default_name` and `name_or_scope` '
                        'is required.')
    g = tf.get_default_graph()
    with g.as_default():
        parent = get_variable_space()
        if name_or_scope is not None:
            if isinstance(name_or_scope, six.string_types):
                name_or_scope = name_or_scope.rstrip('/')
                if parent.name:
                    name_or_scope = parent.name + '/' + name_or_scope
                else:
                    name_or_scope = name_or_scope
                scope_name = name_or_scope
            elif isinstance(name_or_scope, tf.VariableScope):
                name_or_scope = name_or_scope
                scope_name = name_or_scope.name
            else:
                raise TypeError(
                    '`name_or_scope` must be a string or VariableScope.')
            if scope_name:
                with tf.name_scope(scope_name + '/') as ns:
                    assert(ns == scope_name + '/')
                    with _pure_variable_scope(
                            name_or_scope,
                            reuse=reuse,
                            initializer=initializer,
                            regularizer=regularizer,
                            caching_device=caching_device,
                            partitioner=partitioner,
                            custom_getter=custom_getter,
                            dtype=dtype) as vs:
                        yield vs
            else:
                # This can only happen when entering the root variable scope.
                with tf.name_scope(None):
                    with _pure_variable_scope(
                            name_or_scope,
                            reuse=reuse,
                            initializer=initializer,
                            regularizer=regularizer,
                            caching_device=caching_device,
                            partitioner=partitioner,
                            custom_getter=custom_getter,
                            dtype=dtype) as vs:
                        yield vs
        else:
            # Here name_or_scope is None. Using default name, but made unique.
            if reuse:
                raise ValueError(
                    "reuse=True cannot be used without `name_or_scope`.")
            with tf.name_scope(default_name) as scope_name:
                name_or_scope = scope_name.rstrip('/')
                with _pure_variable_scope(
                        name_or_scope,
                        initializer=initializer,
                        regularizer=regularizer,
                        caching_device=caching_device,
                        partitioner=partitioner,
                        custom_getter=custom_getter,
                        dtype=dtype) as vs:
                    yield vs


class ScopedObject(object):
    """Base class with a variable scope.

    Parameters
    ----------
    name : str
        Name of this object, as a part of the variable scope.
        If not specified, will use its parent scope.

    Attributes
    ----------
    name : str
        Name of this object.
    """

    def __init__(self, name=None):
        self._graph = tf.get_default_graph()
        self._parent_scope = get_variable_space()
        if self._parent_scope is None:
            raise RuntimeError('No parent variable scope.')
        self._scope = None  # type: tf.VariableScope
        self.name = name

    @property
    def scope(self):
        """The variable scope of this object."""
        if self._scope is None:
            with self._graph.as_default():
                if self.name is None:
                    self._scope = self._parent_scope
                else:
                    with variable_space(self._parent_scope):
                        with variable_space(None, self.name) as scope:
                            self._scope = scope
        return self._scope

    @contextlib.contextmanager
    def variable_space(self, name=None, reuse=False):
        """Open a variable scope within the scope of this object.

        Parameters
        ----------
        name : str
            Name of the sub variable scope.

            Use the scope of this object if the name is empty or None.

        reuse : bool
            Whether or not to reuse variables in the scope?
        """
        with self._graph.as_default():
            if not name:
                with variable_space(self.scope, reuse=reuse) as scope:
                    yield scope
            else:
                with variable_space(self.scope):
                    with variable_space(name, reuse=reuse) as scope:
                        yield scope
