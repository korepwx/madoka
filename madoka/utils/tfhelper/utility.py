# -*- coding: utf-8 -*-
import contextlib
from logging import getLogger

import tensorflow as tf

__all__ = ['maybe_select_device']


@contextlib.contextmanager
def maybe_select_device(device):
    """Select TensorFlow device if specified.

    Parameters
    ----------
    device : str
        TensorFlow device selector.
        If not None, will select the device for TensorFlow models.
    """
    if device is not None:
        with tf.device(device) as dev:
            getLogger(__name__).info(
                'Selected device %s for TensorFlow.' % device)
            yield dev
    else:
        yield None
