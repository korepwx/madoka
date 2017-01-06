# -*- coding: utf-8 -*-
import os
import shutil

import tensorflow as tf

from madoka.utils import flatten_list, tfhelper, tfcompat
from madoka.utils.tfcompat import scalar_summary, histogram_summary

__all__ = ['SummaryWriter', 'collect_variable_summaries']


class SummaryWriter(object):
    """Wrapper for TensorFlow summary writer.

    Parameters
    ----------
    log_dir : str
        Directory to store the summary files.

    delete_exist : bool
        If ``log_dir`` exists, whether or not to delete all files inside it.
    """

    def __init__(self, log_dir, delete_exist=False):
        if delete_exist and os.path.isdir(log_dir):
            shutil.rmtree(log_dir)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir
        self.tf_writer = tfcompat.SummaryWriter(logdir=log_dir,
                                                graph=tf.get_default_graph())

    def write(self, summary, global_step=None, givens=None):
        """Write a tensor, a variable or a summary object.

        Parameters
        ----------
        summary : tf.Tensor | tf.Variable | tf.Summary | list[tf.Summary]
            Tensor, variable, summary or list of summaries.

        global_step : int
            Global step of this summary.

        givens : dict[str, any]
            Feed dict for session when evaluating ``summary``, if it is
            a tensor or a variable.

        Returns
        -------
        self
        """
        session = tfhelper.ensure_default_session()
        if isinstance(summary, (list, tuple)):
            summary = flatten_list(summary)
        if isinstance(summary, (tf.Tensor, tf.Variable, tf.Operation)):
            summary = session.run(summary, feed_dict=givens)
        self.tf_writer.add_summary(summary, global_step=global_step)
        return self

    def flush(self):
        """Flushes the event file to disk."""
        self.tf_writer.flush()

    def close(self):
        """Flushes the event file to disk and close the writer."""
        self.tf_writer.close()


def collect_variable_summaries(var_or_vars, name=None):
    """Collect the summaries for specified variables.

    Parameters
    ----------
    var_or_vars : Tensor | collections.Iterable[Tensor]
        TensorFlow variable/tensor, or a list of TensorFlow variables/tensors.

    name : str
        Name of the summary operation.

    Returns
    -------
    tf.Summary
        Summary operation for the variables.
    """
    ret = []
    if isinstance(var_or_vars, (tf.Variable, tf.Tensor)):
        var_or_vars = [var_or_vars]
    with tf.name_scope(name):
        for v in var_or_vars:
            name = v.name.split(':', 1)[0]

            # Check the shape of the variable.
            v_shape = v.get_shape()
            v_size = 1
            for s in v_shape.as_list():
                if s is None:
                    v_size = None
                    break
                v_size *= s

            # If there is only one element in the variable, we should just
            # take its value as the summary.
            if v_size is not None and v_size == 1:
                value = v
                for i in range(v_shape.ndims):
                    value = value[0]
                ret.append(scalar_summary(name, value))

            # Otherwise we should gather the statistics of the variable
            else:
                with tf.name_scope(name):
                    ret.append(histogram_summary(name, v))
                    v_mean = tf.reduce_mean(v)
                    v_min = tf.reduce_min(v)
                    v_max = tf.reduce_max(v)
                    with tf.name_scope('stddev_op'):
                        v_stddev = tf.sqrt(tf.reduce_sum(tf.square(v - v_mean)))
                ret.extend([
                    scalar_summary('%s/mean' % name, v_mean),
                    scalar_summary('%s/min' % name, v_min),
                    scalar_summary('%s/max' % name, v_max),
                    scalar_summary('%s/stddev' % name, v_stddev),
                ])
    return ret
