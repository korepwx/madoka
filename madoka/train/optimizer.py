# -*- coding: utf-8 -*-

"""Wrappers for TensorFlow optimizers.

The optimizers provided by TensorFlow might create their own variables
used for training, e.g., the Nesterov Momentum optimizer.  This brings
trouble to model persistence, if these optimizers are to be used as
parameters of models.

Thus this module carries some thin wrappers on the optimizers, which
acts only as factories that creates optimizer instances when needed.
"""

import tensorflow as tf

from madoka.utils import tfhelper

__all__ = [
    'Optimizer', 'TFOptimizerWrapper', 'SGDOptimizer', 'MomentumOptimizer',
    'NesterovMomentumOptimizer', 'AdamOptimizer'
]


class Optimizer(object):
    """Base class for all optimizers."""

    def _create_tf_optimizer(self):
        """Derived classes should override this to create TensorFlow optimizer.

        Returns
        -------
        tf.train.Optimizer
        """
        raise NotImplementedError()

    def minimize(self, loss, params, global_step=None):
        """
        Derivate the update to ``params`` so as to minimize ``loss``.

        Parameters
        ----------
        loss : tf.Tensor
            Tensor expression representing the loss.

        params : collection.Iterable[tf.Tensor]
            List of parameters that should be minimized.

        global_step : tf.Variable
            Global step variable that should be increased.

        Returns
        -------
        tf.Operation
            Operation that updates the parameters.
        """
        optimizer = self._create_tf_optimizer()
        op = optimizer.minimize(loss,
                                var_list=params,
                                global_step=global_step)
        slots = filter(
            lambda v: v,
            (optimizer.get_slot(loss, n)
             for n in optimizer.get_slot_names())
        )
        for v in slots:
            tf.add_to_collection(tfhelper.GraphKeys.TRAINER_SLOTS, v)
        return op

    def maximize(self, loss, params):
        return self.minimize(-loss, params)


class TFOptimizerWrapper(Optimizer):
    """Wraps a TensorFlow optimizer instance as Madoka optimizer."""

    def __init__(self, tf_optimizer):
        self.tf_optimizer = tf_optimizer

    def _create_tf_optimizer(self):
        return self.tf_optimizer


class SGDOptimizer(Optimizer):

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def _create_tf_optimizer(self):
        return tf.train.GradientDescentOptimizer(self.learning_rate)


class MomentumOptimizer(Optimizer):

    def __init__(self, learning_rate=0.001, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum

    def _create_tf_optimizer(self):
        return tf.train.MomentumOptimizer(
            self.learning_rate,
            momentum=self.momentum
        )


class NesterovMomentumOptimizer(Optimizer):

    def __init__(self, learning_rate=0.001, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum

    def _create_tf_optimizer(self):
        return tf.train.MomentumOptimizer(
            self.learning_rate,
            momentum=self.momentum,
            use_nesterov=True
        )


class AdamOptimizer(Optimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def _create_tf_optimizer(self):
        return tf.train.AdamOptimizer(
            self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon
        )
