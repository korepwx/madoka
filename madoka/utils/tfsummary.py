# -*- coding: utf-8 -*-
import os
from collections import deque

import numpy as np
import pandas as pd

__all__ = ['TFSummary', 'find_tf_summary_dirs', 'read_tf_summary_dir']


class TFSummary(object):
    """Data parsed from TensorFlow summary files.

    Parameters
    ----------
    training_loss : pd.Series
        Training loss series, with the step as index.

    validation_loss : pd.Series
        Validation loss series, with the step as index.
    """

    def __init__(self, training_loss=None, validation_loss=None):
        self.training_loss = training_loss
        self.validation_loss = validation_loss

    @classmethod
    def from_accumulator(cls, acc):
        """Extract values from TensorFlow event accumulator.

        Parameters
        ----------
        acc : tensorflow.python.summary.event_accumulator.EventAccumulator
            TensorFlow event accumulator

        Returns
        -------
        TFSummary
        """
        tags = acc.Tags()
        kwargs = {}

        # extract scalar summaries
        def extract_scalar(t):
            events = acc.Scalars(t)
            steps = np.asarray([e.step for e in events], dtype=np.int)
            values = np.asarray([e.value for e in events], dtype=np.float64)
            return pd.Series(index=steps, data=values)

        for tag in tags['scalars']:
            for loss_tag in ('/training_loss', '/validation_loss'):
                if tag.endswith(loss_tag):
                    kwargs[loss_tag[1:]] = extract_scalar(tag)

        # compose the summary object
        return TFSummary(**kwargs)


def find_tf_summary_dirs(root):
    """Find all summary directories from the specified root.

    Directory which contains files of pattern "*.tfevents.*" will be
    considered as a summary directory.

    Parameters
    ----------
    root : str
        Path of the root directory.

    Yields
    ------
    (str, tuple[str])
        A tuple containing the path of summary directory, as well as
        the file names matching "*tfevents*".
    """
    filenames = []
    queue = deque()
    queue.append(root)

    while queue:
        path = queue.popleft()
        for f in os.listdir(path):
            fpath = os.path.join(path, f)
            if os.path.isdir(fpath):
                queue.append(fpath)
            elif 'tfevents' in f:
                filenames.append(f)
        if filenames:
            yield path, filenames
            filenames = []


def read_tf_summary_dir(path):
    """Read summaries from specified directory.

    Parameters
    ----------
    path : str
        Path of the summary directory.

    Returns
    -------
    TFSummary
    """
    from tensorflow.python.summary.event_accumulator import EventAccumulator
    acc = EventAccumulator(path)
    acc.Reload()
    return TFSummary.from_accumulator(acc)
