# -*- coding: utf-8 -*-
"""Convert a set of tensorflow operations into a callable function."""
from collections import OrderedDict

import six
import tensorflow as tf

from .scope import ScopedObject
from .session import ensure_default_session
from ..misc import flatten_list

__all__ = ['Function', 'make_function']


class Function(ScopedObject):
    """A set of TensorFlow operations as a callable function.

    Parameters
    ----------
    inputs : Tensor | list[Tensor] | dict[str, Tensor]
        Given a single tensor, it will be the only unnamed argument.
        Given a list of tensors, they will be the unnamed arguments.
        Given a dict of tensors, they will be the named arguments.

    outputs : Tensor | list[Tensor]
        Output result or results.

    updates : Operation | list[Operation]
        Operation or a list of operations to update variables.

    givens : dict[str, any]
        Feed values to the computation graph.

    name : str
        Name scope of this function.
    """

    def __init__(self, inputs=None, outputs=None, updates=None, givens=None,
                 name='Function'):
        super(Function, self).__init__(name=name)

        # check arguments.
        if inputs is not None and \
                not isinstance(inputs, (dict, OrderedDict, list, tuple)):
            inputs = [inputs]

        if updates is not None:
            if not isinstance(updates, (list, tuple)):
                updates = [updates]
            updates = flatten_list(updates)

        # assign to properties
        self._inputs = inputs
        self._outputs = outputs
        self._updates = updates
        self._givens = givens

        # compile the function
        self._function = self._compile()  # type: (*args, **kwargs) -> any

        # lazy created identity operations
        self._identity_op = {}

    def _create_identity_op(self, v):
        """Create an identity operation for specified variable.

        TensorFlow don't allow us to both feed & fetch the same variable.
        Thus we have to wrap these variables with some computation node.
        """
        if v not in self._identity_op:
            with self.variable_space():
                self._identity_op[v] = tf.identity(v)
        return self._identity_op[v]

    def _compile(self):
        updates = self._updates or []

        if isinstance(self._outputs, (list, tuple)):
            outputs = list(self._outputs)
            fetches = outputs + updates
            direct_value = False
            output_num = len(outputs)
        elif self._outputs is not None:
            fetches = [self._outputs] + updates
            direct_value = True
            output_num = 1
        else:
            fetches = updates
            direct_value = False
            output_num = 0

        if isinstance(self._inputs, (dict, OrderedDict)):
            def get_feed_dict(**kwargs):
                givens = self._givens.copy() if self._givens else {}
                for k, v in six.iteritems(self._inputs):
                    givens[v] = kwargs[k]
                return givens
        else:
            def get_feed_dict(*args):
                givens = self._givens.copy() if self._givens else {}
                if not self._inputs:
                    return givens
                for var, value in zip(self._inputs, args):
                    givens[var] = value
                return givens

        # filter out `None` in fetches, and build function to remap the outputs
        not_null_fetches = [v for v in fetches if v is not None]
        not_null_output_num = len([v for v in fetches[: output_num]
                                   if v is not None])
        fetch_mapping = [i for i, v in enumerate(fetches[: output_num])
                         if v is not None]

        if not_null_output_num == output_num:
            def remap_outputs(values):
                return values
        else:
            def remap_outputs(values):
                ret = [None] * output_num
                for i, v in zip(fetch_mapping, values[: not_null_output_num]):
                    ret[i] = v
                return ret

        def run_func(*args, **kwargs):
            sess = ensure_default_session()
            givens = get_feed_dict(*args, **kwargs)
            fetches2 = [
                self._create_identity_op(v) if v in givens else v
                for v in not_null_fetches
            ]
            ret = sess.run(fetches2, feed_dict=givens)
            ret = remap_outputs(ret[: not_null_output_num])
            return tuple(ret) if not direct_value else ret[0]

        return run_func

    def __call__(self, *args, **kwargs):
        # require there's a session on the stack.
        _ = ensure_default_session()
        args = args or ()
        kwargs = kwargs or {}
        if isinstance(self._inputs, (dict, OrderedDict)):
            if args:
                raise ValueError('Function only accepts named arguments.')
            for k, v in six.iteritems(kwargs):
                if k not in self._inputs:
                    raise ValueError('Unexpected named argument %s.' % k)
            for k, v in six.iteritems(self._inputs):
                if k not in kwargs:
                    raise ValueError('Named argument %s is required but not '
                                     'specified.' % k)
        else:
            if kwargs:
                raise ValueError('Function only accepts unnamed arguments.')
            if len(args) != len(self._inputs or ()):
                raise ValueError('Require %d unnamed arguments, but got %s.' %
                                 (len(self._inputs or ()), len(args)))

        return self._function(*args, **kwargs)


def make_function(inputs=None, outputs=None, updates=None, givens=None,
                  name=None):
    """Bundle TensorFlow operations into a callable function.

    Parameters
    ----------
    inputs : Tensor | list[Tensor] | dict[str, Tensor]
        Given a single tensor, it will be the only unnamed argument.
        Given a list of tensors, they will be the unnamed arguments.
        Given a dict of tensors, they will be the named arguments.

    outputs : Tensor | list[Tensor]
        Output result or results.

    updates : Operation | list[Operation]
        Operation or a list of operations to update variables.

    givens : dict[str, any]
        Feed values to the computation graph.

    name : str
        Name scope of this function.
    """
    return Function(inputs=inputs, outputs=outputs, updates=updates,
                    givens=givens, name=name)
