# -*- coding: utf-8 -*-
import tensorflow as tf

from ..graph import GraphKeys
from ...tfcompat import global_variables

__all__ = ['VariableSelectorContext', 'VariableSelector']


class VariableSelectorContext(object):
    """Context for selecting variables.

    Parameters
    ----------
    graph : tf.Graph
        TensorFlow graph.

    all_variables : collections.Iterable[tf.Variable]
        Whole set of variables.
    """

    def __init__(self, graph, all_variables):
        self.graph = graph
        self.all_variables = set(all_variables)

    @classmethod
    def for_graph(cls, graph=None):
        """Get a variable selector context for graph.

        All the variables in the specified graph will be chosen as the
        complete set of the selector context.

        Parameters
        ----------
        graph : tf.Graph
            TensorFlow graph.

        Returns
        -------
        VariableSelectorContext
        """
        graph = graph or tf.get_default_graph()
        return VariableSelectorContext(graph, global_variables())


class VariableSelector(object):
    """Logical term of variable selector.

    You may use "+", "-", "&" and "|" to make composition of the selectors,
    whose resulting set of variables should be:

        `a & b`: intersection of `a` and `b`.
        `a | b`: union of `a` and `b`.
        `a + b`: same as `a | b`.
        `a - b`: a & (-b)
        `-a`: complimentary of `a` (against `ctx.all_variables`).
    """

    def select(self):
        """Collect variables according to this selector.

        Returns
        -------
        list[tf.Variable]
        """
        ctx = VariableSelectorContext.for_graph()
        variables = self._select(ctx)
        variables.intersection_update(ctx.all_variables)
        return list(variables)

    def _select(self, ctx):
        raise NotImplementedError()

    def __and__(self, other):
        if isinstance(other, _NotSelector):
            ret = _AndSelector([self], [other._selector])
        else:
            ret = _AndSelector([self, other])
        return ret

    def __or__(self, other):
        return _OrSelector([self, other])

    def __neg__(self):
        return _NotSelector(self)

    def __add__(self, other):
        return self | other

    def __sub__(self, other):
        return self & (-other)

    @classmethod
    def all(cls):
        """Select all variables."""
        return _AllSelector()

    @classmethod
    def list(cls, variables):
        """Select the specified variables.

        Parameters
        ----------
        variables : tf.Variable | collections.Iterable[tf.Variables]
            Variable or list of variables to be selected.
        """
        return _ListSelector(variables)

    @classmethod
    def collection(cls, key):
        """Select the variables in specified collection.

        Parameters
        ----------
        key : str
            Name of the collection.
        """
        return _CollectionSelector(key)

    @classmethod
    def trainable(cls):
        """Select trainable variables."""
        return _TrainableSelector()

    @classmethod
    def training_states(cls):
        """Select training states variables."""
        return _TrainingStateSelector()

    @classmethod
    def trainer_slots(cls):
        """Select trainer slots variables."""
        return _TrainerSlotSelector()

    @classmethod
    def scope(cls, scope):
        """Select the variables in specified scope.

        Parameters
        ----------
        scope : str
            Name of the scope
        """
        return _ScopeScelector(scope)


class _AndSelector(VariableSelector):

    def __init__(self, base, exclude=None):
        base = list(base)
        exclude = list(exclude) if exclude else []
        if not base:
            raise ValueError('No `include` selector.')
        self._base = base
        self._exclude = exclude

    def _select(self, ctx):
        ret = self._base[0]._select(ctx)
        for s in self._base[1:]:
            ret.intersection_update(s._select(ctx))
        for s in self._exclude:
            ret.difference_update(s._select(ctx))
        return ret

    def __and__(self, other):
        if isinstance(other, _AndSelector):
            ret = _AndSelector(self._base + other._base,
                               self._exclude + other._exclude)
        elif isinstance(other, _NotSelector):
            ret = _AndSelector(self._base, self._exclude + [other._selector])
        else:
            ret = _AndSelector(self._base + [other], self._exclude)
        return ret

    def __repr__(self):
        return '(%s)' % ' & '.join([repr(s) for s in self._base] +
                                   ['(-%s)' % s for s in self._exclude])


class _OrSelector(VariableSelector):

    def __init__(self, selectors):
        selectors = list(selectors)
        if not selectors:
            raise ValueError('No selector.')
        self._selectors = selectors

    def _select(self, ctx):
        ret = self._selectors[0]._select(ctx)
        for s in self._selectors[1:]:
            ret.update(s._select(ctx))
        return ret

    def __or__(self, other):
        if isinstance(other, _OrSelector):
            ret = _OrSelector(self._selectors + other._selectors)
        else:
            ret = _OrSelector(self._selectors + [other])
        return ret

    def __repr__(self):
        return '(%s)' % ' + '.join(repr(s) for s in self._selectors)


class _NotSelector(VariableSelector):

    def __init__(self, selector):
        self._selector = selector

    def _select(self, ctx):
        return ctx.all_variables.difference(self._selector._select(ctx))

    def __and__(self, other):
        if isinstance(other, _AndSelector):
            ret = other & self
        else:
            ret = super(_NotSelector, self).__and__(other)
        return ret

    def __neg__(self):
        return self._selector

    def __repr__(self):
        return '(-%s)' % self._selector


class _AllSelector(VariableSelector):

    def _select(self, ctx):
        return ctx.all_variables

    def __repr__(self):
        return 'all'


class _ListSelector(VariableSelector):

    def __init__(self, variables):
        if isinstance(variables, tf.Variable):
            variables = {variables}
        else:
            variables = set(variables)
        self._variables = variables

    def _select(self, ctx):
        return self._variables

    def __repr__(self):
        return '{%s}' % ','.join(sorted(v.name for v in self._variables))


class _CollectionSelector(VariableSelector):
    KEY = None

    def __init__(self, key=None):
        self._key = key or self.KEY

    def _select(self, ctx):
        return set(ctx.graph.get_collection(self._key))

    def __repr__(self):
        return self._key


class _TrainableSelector(_CollectionSelector):
    KEY = tf.GraphKeys.TRAINABLE_VARIABLES


class _TrainingStateSelector(_CollectionSelector):
    KEY = GraphKeys.TRAINING_STATES


class _TrainerSlotSelector(_CollectionSelector):
    KEY = GraphKeys.TRAINER_SLOTS


class _ScopeScelector(VariableSelector):

    def __init__(self, scope):
        scope = scope.rstrip('/') + '/'
        self._scope = scope

    def _select(self, ctx):
        return set(v for v in ctx.all_variables
                   if v.name.startswith(self._scope))

    def __repr__(self):
        return '@(%s)' % self._scope
