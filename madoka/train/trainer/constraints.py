# -*- coding: utf-8 -*-
"""Trainer constraint classes."""

__all__ = ['WeightedTrainer', 'EnsembleClassifierBuilder']


class WeightedTrainer(object):
    """Constraint for trainers that support data weights."""

    def set_weight(self, weight):
        """Set the weight of data.

        Parameters
        ----------
        weight : numpy.ndarray | tf.Variable | tf.Tensor
            The weight of training data.
        """
        raise NotImplementedError()


class EnsembleClassifierBuilder(object):
    """Constraint for trainers that can build ensemble classifiers."""

    def ensemble_classifier(self, input_ph):
        """Build a classifier upon the ensemble models.

        This method is only guaranteed to work after the ensemble models
        have already been trained.

        Parameters
        ----------
        input_ph : tf.Tensor | collections.Iterable[tf.Tensor]
            The input placeholders, except label.

        Returns
        -------
        Classifier
        """
        raise NotImplementedError()
