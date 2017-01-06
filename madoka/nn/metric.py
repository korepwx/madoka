# -*- coding: utf-8 -*-

__all__ = ['ErrorMetric', 'SquareError']


class ErrorMetric(object):
    """Base class for all error metrics.

    An error metric should accept two tensors of same shape, and compute a
    non-negative tensor indicating the "element-wise" distance between these
    two tensors.

    The shape of the output tensor usually matches the input tensors, however,
    in some rare cases this might not hold.  If is free for the error metric
    to choose the meaning of "element-wise" distance.
    It is only guaranteed that the first dimension, representing different
    data points, must be identical with the input tensors.
    """

    def __call__(self, a, b):
        """Compute the distance between ``a`` and ``b``.

        Parameters
        ----------
        a, b : tf.Tensor | tf.Variable
            Tensors whose distance should be computed.

        Returns
        -------
        tf.Tensor
            Non-negative tensor indicating the "element-wise" distance
            between the two input tensors.
        """
        return self.distance(a, b)

    def distance(self, a, b):
        raise NotImplementedError()


class SquareError(ErrorMetric):
    """Square error between two tensors."""

    def distance(self, a, b):
        return (a - b) ** 2
