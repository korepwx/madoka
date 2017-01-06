# -*- coding: utf-8 -*-
import unittest

import numpy as np
import tensorflow as tf

from madoka.dataflow import DataFlow
from madoka.train import CheckpointMonitor, ValidationMonitor, FixedMaxSteps
from madoka.utils import TemporaryDirectory, tfhelper
from madoka.utils.tfcompat import global_variables, global_variables_initializer
from madoka.utils.tfhelper import VariableSetter


class MonitorTestCase(unittest.TestCase):
    """Monitor unit tests."""

    def test_checkpoint_monitor(self):
        """CheckpointMonitor unit test."""
        with tf.Graph().as_default():
            a = tf.get_variable('a', dtype=tf.int32, initializer=0)
            a_setter = VariableSetter(a)

            with TemporaryDirectory() as tmpdir:
                m = CheckpointMonitor(tmpdir, global_variables(), steps=2)
                init_op = global_variables_initializer()

                # test the checkpoint monitor could actually restore variables
                with tf.Session() as sess:
                    sess.run(init_op)
                    self.assertEqual(tfhelper.get_variable_values(a), 0)
                    m.before_training()
                    self.assertEqual(tfhelper.get_variable_values(a), 0)
                    m.start_training(64, 100, max_steps=FixedMaxSteps(100),
                                     initial_step=0)
                    a_setter.set(1)
                    m.end_step(2, 0.0)
                    self.assertEqual(tfhelper.get_variable_values(a), 1)
                    a_setter.set(2)
                    m.end_step(3, 0.0)
                    self.assertEqual(tfhelper.get_variable_values(a), 2)
                    m.end_training()

                with tf.Session() as sess:
                    sess.run(init_op)
                    self.assertEqual(tfhelper.get_variable_values(a), 0)
                    m.before_training()
                    self.assertEqual(tfhelper.get_variable_values(a), 1)
                    m.start_training(64, 100, max_steps=FixedMaxSteps(100),
                                     initial_step=0)
                    m.end_training()

    def test_validation_monitor_large_loss(self):
        """Validation monitor unit test with extremely large loss."""
        class DummyValidator(object):

            def __init__(self, dtype=np.float32):
                self.losses = np.asarray(
                    [
                        # ValidationMonitor will not do validation at step 0,
                        # so these losses should belong to step 1, 2, 3, ...
                        10296686154590674.0,
                        0.0,
                        4606703541986263.0,
                    ],
                    dtype=dtype
                )
                self.loss_id = 0

            def __call__(self, *args, **kwargs):
                ret = self.losses[self.loss_id]
                self.loss_id += 1
                return ret

        def do_test(dtype=np.float32):
            valid_fn = DummyValidator(dtype=dtype)
            with tf.Graph().as_default():
                a = tf.get_variable('a', dtype=tf.int32, initializer=-1,
                                    trainable=True)
                a_setter = VariableSetter(a)
                max_steps = FixedMaxSteps(100)
                init_op = global_variables_initializer()

                with TemporaryDirectory() as tmpdir:
                    m = ValidationMonitor(
                        valid_fn,
                        DataFlow.from_numpy(np.arange(3)),
                        tmpdir,
                        [a],
                        steps=1
                    )

                    # test the validation monitor can handle large losses
                    with tf.Session() as sess:
                        sess.run(init_op)
                        self.assertEquals(tfhelper.get_variable_values(a), -1)
                        max_steps.init_training()
                        m.before_training()
                        self.assertEquals(tfhelper.get_variable_values(a), -1)
                        m.start_training(64, 100, max_steps=max_steps,
                                         initial_step=0)
                        m.start_epoch(0)
                        for i in range(len(valid_fn.losses)):
                            a_setter.set(i)
                            m.end_step(i, 0.0)
                            self.assertEquals(tfhelper.get_variable_values(a),
                                              i)
                        m.end_epoch(0, 0.0)
                        m.end_training()
                        self.assertEquals(tfhelper.get_variable_values(a), 2)

        do_test(np.float32)
        do_test(np.float64)
