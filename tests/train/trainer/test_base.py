# -*- coding: utf-8 -*-
import unittest

import numpy as np
import tensorflow as tf

from madoka.dataflow import DataFlow
from madoka.train import Trainer


class MyTrainer(Trainer):
    def __init__(self, name='MyTrainer', **kwargs):
        super(MyTrainer, self).__init__(name=name, **kwargs)
        self._my_train_flow = None
        self._my_valid_flow = None

    def _run(self, checkpoint_dir, train_flow, valid_flow):
        self._my_train_flow = train_flow
        self._my_valid_flow = valid_flow


class BaseTrainerTestCase(unittest.TestCase):
    """Unit tests for base trainers."""

    def test_basic(self):
        """Test the basic interfaces of a trainer."""
        with tf.Graph().as_default():
            v1 = tf.get_variable(
                'v1', initializer=0, dtype=tf.int32, trainable=True)
            v2 = tf.get_variable(
                'v2', initializer=0, dtype=tf.int32, trainable=False)

            t = MyTrainer(max_steps=1, validation_split=.1)
            self.assertEquals(t.trainable_vars.select(), [v1])

            self.assertIsNone(t.placeholders)
            ph = tf.placeholder(tf.int32, shape=(), name='ph')
            t.set_placeholders(ph)
            self.assertEquals(t.placeholders, [ph])

            df = DataFlow.from_numpy(np.arange(100))
            t.set_data_flow(df, shuffle=True)
            with tf.Session():
                t.run()

            self.assertEquals(len(t._my_train_flow), 90)
            self.assertEquals(len(t._my_valid_flow), 10)
            arr = np.concatenate([t._my_train_flow.all()[0],
                                  t._my_valid_flow.all()[0]], axis=0)
            np.testing.assert_array_equal(sorted(arr), df.all()[0])
            self.assertFalse(np.all(arr == df.all()[0]))


if __name__ == '__main__':
    unittest.main()
