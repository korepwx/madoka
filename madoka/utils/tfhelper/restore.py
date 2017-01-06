# -*- coding: utf-8 -*-
import os
from logging import getLogger

import tensorflow as tf

from .session import ensure_default_session

__all__ = ['SessionRestorer']


class SessionRestorer(object):
    """Class to save and restore TensorFlow sessions.

    Parameters
    ----------
    variables : collections.Iterable[tf.Variable]
        List of variables, which should be saved and restored.

    path : str
        Path of the checkpoint file.

    max_versions : int
        Maximum versions to keep in the directory.

        At least 2 versions should be kept, in order to prevent corrupted
        checkpoint files caused by IO failure.

    latest_file : str
        Name of the file which organizes the checkpoint versions.

    save_meta : bool
        Whether or not to save meta graph.  Default is True.

    name : str
        Name of this session restorer.
    """

    def __init__(self, variables, path, max_versions=2, latest_file='latest',
                 save_meta=True, name='SessionRestorer'):
        variables = list(variables)
        for v in variables:
            if not isinstance(v, tf.Variable):
                raise TypeError('%r is not a TensorFlow variable.' % (v,))
        if max_versions < 2:
            raise ValueError('At least 2 versions should be kept.')
        self.variables = variables
        self.path = os.path.abspath(path)
        self.max_versions = max_versions
        self.latest_file = latest_file
        self.save_meta = save_meta
        self.name = name
        self._dirpath, self._filename = os.path.split(self.path)
        self._saver = self._build_saver()

    def _build_saver(self):
        return tf.train.Saver(var_list=self.variables,
                              max_to_keep=self.max_versions,
                              name=self.name)

    def get_latest_file(self):
        """Get the latest available checkpoint file."""
        return tf.train.latest_checkpoint(self._dirpath, self.latest_file)

    def save(self, global_step=None):
        """Save the checkpoint to file.

        Parameters
        ----------
        global_step : int | tf.Tensor
            The global step counter.
        """
        sess = ensure_default_session()
        if not os.path.isdir(self._dirpath):
            os.makedirs(self._dirpath)
        self._saver.save(
            sess,
            self.path,
            global_step=global_step,
            latest_filename=self.latest_file,
            write_meta_graph=self.save_meta
        )

    def restore(self):
        """Restore the checkpoint from file if it exists."""
        chkpath = self.get_latest_file()
        if chkpath:
            sess = ensure_default_session()
            getLogger(__name__).info(
                'Restore from checkpoint file %r.',
                chkpath
            )
            self._saver.restore(sess, chkpath)
