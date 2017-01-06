# -*- coding: utf-8 -*-
from logging import getLogger

import numpy as np
import six
import tensorflow as tf
from sklearn.base import BaseEstimator

from madoka.train import Monitor, EnsembleTrainer
from madoka.utils import flatten_list
from madoka.utils.tfhelper import ensure_default_session

if six.PY2:
    from itertools import izip_longest as zip_longest
else:
    from itertools import zip_longest


class Event(object):
    def __init__(self, name, args):
        self.name = name
        self.args = sorted(six.iteritems(args))

    def __repr__(self):
        if self.args:
            args_repr = ','.join('%s=%s' % a for a in self.args)
            return '%s:%s' % (self.name, args_repr)
        else:
            return self.name


class EventCollector(object):

    def __init__(self):
        self._events = []

    def add(self, e):
        self._events.append(e)

    def dump(self):
        return '\n'.join(repr(r) for r in self._events)

    def match(self, patterns):
        mismatch = False
        marks = []

        for i, (e, p) in enumerate(zip(self._events, patterns)):
            if isinstance(p, six.string_types):
                not_match = p != repr(e)
            elif hasattr(p, 'match'):
                not_match = not p.match(repr(e))
            elif callable(p):
                not_match = not p(e)
            else:
                raise TypeError('%r is not a pattern.' % (p,))
            if not_match:
                mismatch = True
                marks.append('[x] ')
            else:
                marks.append('    ')
        if len(self._events) > len(patterns):
            mismatch = True
            marks += ['[+] '] * (len(self._events) - len(patterns))
        if len(self._events) < len(patterns):
            mismatch = True
            marks += ['[-] '] * (len(patterns) - len(self._events))
        if mismatch:
            msg = [
                'Events mismatch:',
                '  expected events:',
                '\n'.join('    %s' %
                          (e.pattern if hasattr(e, 'pattern') else e)
                          for e in patterns),
                '  actual events:',
                '\n'.join('%s%s' % (m, e or '(null)')
                          for e, m in zip_longest(self._events, marks))
            ]
            raise AssertionError('\n'.join(msg))

    def __iter__(self):
        return iter(self._events)

    def __len__(self):
        return len(self._events)


class MonitorEventLogger(Monitor):

    def __init__(self, name='MonitorEventLogger', break_after_steps=None,
                 collector=None):
        super(MonitorEventLogger, self).__init__(name)
        self.events = EventCollector() if collector is None else collector
        self.break_after_steps = break_after_steps

    def add_event(self, name, **kwargs):
        self.events.add(Event(name, kwargs))

    def before_training(self):
        self.add_event('before_training')

    def start_training(self, batch_size, epoch_steps, max_steps, initial_step):
        self.add_event(
            'start_training', batch_size=batch_size, epoch_steps=epoch_steps,
            max_steps=max_steps, initial_step=initial_step
        )

    def end_training(self, has_error=False):
        """Notify the monitor that a training process has finished.

        It will be triggered whether or not any error has taken place.

        Parameters
        ----------
        has_error : bool
            Whether or not any error has occurred during training.
        """
        self.add_event('end_training', has_error=has_error)

    def start_epoch(self, epoch):
        """Notify the monitor that a training epoch will start.

        Parameters
        ----------
        epoch : int
            Index of the epoch, starting from 0.
        """
        self.add_event('start_epoch', epoch=epoch)

    def end_epoch(self, epoch, avg_loss):
        """Notify the monitor that a training epoch has completed.

        Parameters
        ----------
        epoch : int
            Index of the epoch, starting from 0.

        avg_loss : float
            Average training loss of all steps in this epoch.
            Would be None if the training process does not evolve a loss.
        """
        self.add_event('end_epoch', epoch=epoch, avg_loss=avg_loss)

    def start_step(self, step):
        """Notify the monitor that a training step (mini-batch) will start.

        Parameters
        ----------
        step : int
            Index of the step, starting from 0.

            This should be the total number of steps have ever been performed
            since the whole training process started, not from the start of
            this epoch.
        """
        self.add_event('start_step', step=step)

    def end_step(self, step, loss):
        """Notify the monitor that a training step (mini-batch) has completed.

        Parameters
        ----------
        step : int
            Index of the step, starting from 0.

        loss : float
            Training loss of this step.
            Would be None if the training process does not evolve a loss.
        """
        self.add_event('end_step', step=step, loss=loss)
        # Note that `step` counts from zero.
        if self.break_after_steps is not None \
                and self.break_after_steps <= step + 1:
            raise KeyboardInterrupt()


class SummaryEvent(Event):

    def __init__(self, step, summary):
        s = tf.Summary()
        s.ParseFromString(summary)
        values = {}
        for val in s.value:
            values[val.tag] = val.simple_value
        super(SummaryEvent, self).__init__(step, values)


class SummaryWriterLogger(object):

    def __init__(self, collector=None):
        self.events = EventCollector() if collector is None else collector

    def write(self, summary, global_step=None, givens=None):
        session = ensure_default_session()
        if isinstance(summary, (list, tuple)):
            summary = flatten_list(summary)
        if isinstance(summary, (tf.Tensor, tf.Variable)):
            summary = session.run(summary, feed_dict=givens)
        self.events.add(SummaryEvent(global_step, summary))
        return self


class _EnsembleTrainerMixin:

    def _init_mixin(self, collector=None):
        self.events = EventCollector() if collector is None else collector

        def make_wrapper(method, logger):
            @six.wraps(method)
            def inner(*args, **kwargs):
                logger(*args, **kwargs)
                return method(*args, **kwargs)
            return inner

        for k in ('_prepare_data_flow_for_child',
                  '_before_training', '_after_training',
                  '_before_child_training', '_after_child_training'):
            mk = '_m' + k
            setattr(self, k, make_wrapper(getattr(self, k), getattr(self, mk)))

    def add_event(self, name, **kwargs):
        self.events.add(Event(name, kwargs))

    def _m_prepare_data_flow_for_child(self, model_id, train_flow):
        self.add_event('prepare_data_flow_for_child', model_id=model_id)

    def _m_before_training(self, initial_model_id):
        self.add_event('before_training', initial_model_id=initial_model_id)

    def _m_after_training(self):
        self.add_event('after_training')

    def _m_before_child_training(self, model_id):
        self.add_event('before_child_training', model_id=model_id)

    def _m_after_child_training(self, model_id):
        self.add_event('after_child_training', model_id=model_id)


class EnsembleTrainerWithLogger(EnsembleTrainer, _EnsembleTrainerMixin):

    def __init__(self, collector=None, **kwargs):
        super(EnsembleTrainerWithLogger, self).__init__(**kwargs)
        self._init_mixin(collector=collector)


class BinarySplitter(BaseEstimator):

    def __init__(self, boundary=.0):
        self.boundary = boundary
        self.classes_ = np.asarray([0, 1], dtype=np.int32)

    def fit(self, X, y, sample_weight=None):
        getLogger(__name__).debug(
            list(np.asarray(sample_weight, dtype=np.float32)))
        return self

    def predict(self, X):
        return np.asarray(X >= 0., dtype=np.int32).reshape([-1])

    def predict_proba(self, X):
        p = 1. / (1 + np.exp(-X))
        return np.concatenate([1 - p, p], axis=1)


class MultimodalSplitter(BaseEstimator):

    def __init__(self, boundaries):
        self.boundaries = np.asarray(boundaries, dtype=np.float32)
        self.middles = (self.boundaries[1:] + self.boundaries[:-1]) * .5
        self.distances = self.boundaries[1:] - self.boundaries[:-1]
        self.classes_ = np.arange(len(boundaries)+1, dtype=np.int32)

    def fit(self, X, y, sample_weight=None):
        getLogger(__name__).debug(
            list(np.asarray(sample_weight, dtype=np.float32)))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def _make_proba_components(self, X, sigmoid, select):
        proba = []

        # the first component
        left = self.boundaries[0] - self.distances[0] * .5
        proba.append(sigmoid((left - X) / self.distances[0] * 10.))

        # the middle components
        for b, d in zip(self.boundaries, self.distances):
            m = b + d * .5
            proba.append(
                select(
                    X < m,
                    sigmoid((X - m) / d * 10.),
                    sigmoid((m - X) / d * 10.)
                )
            )

        # the last component
        right = self.boundaries[-1] + self.distances[-1] * .5
        proba.append(sigmoid((X - right) / self.distances[-1] * 10.))

        return proba

    def predict_proba(self, X):
        def sigmoid(x):
            return 1./(1.+np.exp(-x))

        def select(cond, x, y):
            cond = np.asarray(cond, dtype=np.int32)
            return cond * x + (1 - cond) * y

        if len(X.shape) == 2:
            X = X.reshape([-1])
        proba = self._make_proba_components(X, sigmoid, select)

        # gather and normalize the probability
        for i in range(len(proba)):
            proba[i] = proba[i].reshape([-1, 1])
        proba = np.concatenate(proba, axis=1)
        proba = proba / np.sum(proba, axis=1).reshape([-1, 1])
        return proba


def make_multimodal_tf_splitter(boundaries, x):
    sp = MultimodalSplitter(boundaries)
    proba = sp._make_proba_components(x, tf.sigmoid, tf.select)

    # gather and normalize the probability
    for i in range(len(proba)):
        proba[i] = tf.reshape(proba[i], [-1, 1])
    proba = tf.concat(1, proba)
    proba = proba / tf.reduce_sum(proba, [1], keep_dims=True)
    return proba

