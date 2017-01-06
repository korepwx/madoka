# -*- coding: utf-8 -*-
import contextlib
import copy
import threading

import six

from madoka.utils import TrainingPhase
from .base import DataFlow

__all__ = ['DataFlowFlags', 'DataFlowContext', 'get_dataflow_context']


class DataFlowFlags(object):
    """Flags for data flow context."""

    phase = TrainingPhase.NOT_SET
    """The training phase."""

    def __init__(self, **kwargs):
        self.set_flags(**kwargs)

    @property
    def is_training(self):
        """Whether or not it is in training phase."""
        return self.phase == TrainingPhase.TRAINING

    @is_training.setter
    def is_training(self, value):
        if value not in (TrainingPhase.NOT_SET, TrainingPhase.TRAINING,
                         TrainingPhase.VALIDATION, TrainingPhase.TESTING):
            raise ValueError('%r is not a valid training phase.' % (value,))
        self.phase = value

    def as_dict(self):
        """Get flags as dict.

        Returns
        -------
        dict[str, any]
            Dict of flags.
        """
        return {k: v for k, v in six.iteritems(self.__dict__)}

    def set_flags(self, **kwargs):
        """Set flag values.

        Parameters
        ----------
        **kwargs : dict[str, any]
            Dict of flags.
        """
        for k, v in six.iteritems(kwargs):
            setattr(self, k, v)

    def merge_from(self, other):
        """Merge flags from another DataFlowFlags object.

        Parameters
        ----------
        other : DataFlowFlags
            Another flags object.
        """
        self.set_flags(**other.as_dict())


class DataFlowContext(object):
    """Data flow context.

    Some data flows might determine its behaviour according to the context.
    For example, a noise data flow may only add noises to underlying data
    flow at training time.  This class thus represents the data flow context,
    and can be pushed to a per-thread stack using `as_default()`.

    The second function of a data flow context is to gather all the ancestor
    data flows given any of the terminal flows, and to call `reset_epoch()`
    of these flows in the correct order.

    Parameters
    ----------
    flows : DataFlow | collections.Iterable[DataFlow]
        Terminal or intermediate data flows.

    flags : DataFlowFlags
        Initial values of flags.
        A copy will be made to avoid overriding on this object.
    """

    def __init__(self, flows, flags=DataFlowFlags()):
        if isinstance(flows, DataFlow):
            flows = (flows,)
        else:
            flows = tuple(flows)
        for flow in flows:
            if not isinstance(flow, DataFlow):
                raise TypeError('%r is not a data flow.' % (flow,))
        all_flows = self._topological_sort(flows)

        self.flags = copy.copy(flags)
        self.flows = flows
        self.all_flows = all_flows

    @staticmethod
    def _topological_sort(flows):
        ret = []
        discovered = set()

        def dps(flow, chain=()):
            if flow not in discovered:
                if flow in chain:
                    raise RuntimeError('Data flow cycle discovered: %r.' %
                                       chain)
                chain += (flow,)
                discovered.add(flow)
                for f in flow.input_flows:
                    dps(f, chain)
                ret.append(flow)

        for child in flows:
            dps(child)
        return tuple(ret)

    def reset_epoch(self):
        """Let all data flows prepare for next epoch in the correct order."""
        for f in self.all_flows:
            f.reset_epoch()

    @contextlib.contextmanager
    def as_default(self):
        """Open a scope and push this context to the top of thread stack.

        The context will be removed from the stack after exiting the scope.
        """
        _ctx_stack.push(self)
        try:
            yield self
        finally:
            _ctx_stack.pop()

    @contextlib.contextmanager
    def set_flags(self, **kwargs):
        """Open a scope and set context flags.

        The flags will be restored after exiting the scope.

        Parameters
        ----------
        **kwargs
            Flags and their values.
        """
        # update the flags and memorize all changes
        added, updated = [], {}
        for k, v in six.iteritems(kwargs):
            if hasattr(self.flags, k):
                updated[k] = getattr(self.flags, k)
            else:
                added.append(k)
            setattr(self.flags, k, v)

        try:
            yield self
        finally:
            # restore the flags
            for k, v in six.iteritems(updated):
                setattr(self.flags, k, v)
            for k in added:
                delattr(self.flags, k)


def get_dataflow_context():
    """Get the data flow context at the top of context stack.

    A data flow context can be pushed to the top of stack by calling
    `as_default()` method.

    Returns
    -------
    DataFlowContext
        The data flow context at the top.
    """
    return _ctx_stack.top()


class _ContextStack(object):
    _STORAGE_STACK_KEY = '_context_stack'

    def __init__(self):
        self._storage = threading.local()

    @property
    def items(self):
        if not hasattr(self._storage, self._STORAGE_STACK_KEY):
            setattr(self._storage, self._STORAGE_STACK_KEY, [])
        return getattr(self._storage, self._STORAGE_STACK_KEY)

    def push(self, context):
        self.items.append(context)

    def pop(self):
        self.items.pop()

    def top(self):
        items = self.items
        if items:
            return items[-1]

_ctx_stack = _ContextStack()
