# -*- coding: utf-8 -*-
import sys
import unittest

import numpy as np
import tensorflow as tf

from madoka.nn import dropout
from madoka.utils import tfhelper, TrainingPhase
from madoka.utils.tfcompat import global_variables_initializer
from madoka.utils.tfhelper import Bookkeeper

try:
    if sys.platform != 'win32':
        # Currently TFLearn does not support win32.
        import tflearn
    else:
        tflearn = None
except ImportError:
    tflearn = None


class IntegrationTestCase(unittest.TestCase):
    """Unit test cases for TensorFlow integration utilities."""

    def test_training_phase(self):
        """Test the training phase."""
        with tf.Graph().as_default():
            books = Bookkeeper.for_graph()
            flag = books.training_phase
            phase = flag.phase_tensor
            is_training = flag.is_training_tensor

            with tf.Session() as sess:
                sess.run(global_variables_initializer())
                self.assertEquals(sess.run(phase), TrainingPhase.NOT_SET)
                self.assertFalse(sess.run(is_training))

                with tfhelper.set_training_phase(TrainingPhase.TRAINING):
                    self.assertEquals(sess.run(phase), TrainingPhase.TRAINING)
                    self.assertTrue(sess.run(is_training))

                    with tfhelper.set_training_phase(TrainingPhase.VALIDATION):
                        self.assertEquals(sess.run(phase),
                                          TrainingPhase.VALIDATION)
                        self.assertFalse(sess.run(is_training))

                    with tfhelper.set_training_phase(TrainingPhase.TESTING):
                        self.assertEquals(sess.run(phase),
                                          TrainingPhase.TESTING)
                        self.assertFalse(sess.run(is_training))

                    with tfhelper.set_training_phase(TrainingPhase.NOT_SET):
                        self.assertEquals(sess.run(phase),
                                          TrainingPhase.NOT_SET)
                        self.assertFalse(sess.run(is_training))

                    self.assertEquals(sess.run(phase), TrainingPhase.TRAINING)
                    self.assertTrue(sess.run(is_training))

            # test training phase with a dropout layer
            N = 10000
            p = .3
            data = np.ones([N, 1], dtype=np.float32)
            input_ph = tfhelper.make_placeholder_for('input', data)
            output_tensor = dropout(input_ph, keep_prob=p)
            f = tfhelper.make_function(input_ph, output_tensor)

            with tf.Session() as sess:
                sess.run(global_variables_initializer())
                with tfhelper.set_training_phase(TrainingPhase.TRAINING):
                    output = f(data)
                    self.assertFalse(np.all(data == output))
                    output_mean = np.average(output)

                    # According to large number theory, the mean error follows
                    # N(0, sqrt(p(1-p)/N)).  Thus the following statement should
                    # be satisfied with a very high probability.
                    self.assertLess(np.abs(output_mean * p - p),
                                    3 * np.sqrt(p * (1 - p) / N))

                    # Dropout should be a binomial distribution * (v/keep_prob)
                    self.assertTrue(
                        np.all(
                            (np.abs(output - data / p) < 1e-5) |
                            (np.abs(output) < 1e-5)
                        )
                    )

                    with tfhelper.set_training_phase(TrainingPhase.NOT_SET):
                        output = f(data)
                        np.testing.assert_array_equal(data, output)

    @unittest.skipIf(tflearn is None, 'TFLearn is not installed.')
    def test_tflearn_training_flag(self):
        """Test the training flag of TFLearn."""
        with tf.Graph().as_default():
            N = 10000
            p = .3
            data = np.ones([N, 1], dtype=np.float32)
            input_ph = tfhelper.make_placeholder_for('input', data)
            output_tensor = tflearn.dropout(input_ph, keep_prob=p)
            f = tfhelper.make_function(input_ph, output_tensor)
            _ = Bookkeeper.for_graph()      # ensure flags are created

            with tf.Session() as sess:
                sess.run(global_variables_initializer())
                with tfhelper.set_training_phase(TrainingPhase.TRAINING):
                    output = f(data)
                    self.assertFalse(np.all(data == output))
                    output_mean = np.average(output)

                    # According to large number theory, the mean error follows
                    # N(0, sqrt(p(1-p)/N)).  Thus the following statement should
                    # be satisfied with a very high probability.
                    self.assertLess(np.abs(output_mean * p - p),
                                    3 * np.sqrt(p * (1 - p) / N))

                    # Dropout should be a binomial distribution * (v/keep_prob)
                    self.assertTrue(
                        np.all(
                            (np.abs(output - data / p) < 1e-5) |
                            (np.abs(output) < 1e-5)
                        )
                    )

                    with tfhelper.set_training_phase(TrainingPhase.NOT_SET):
                        output = f(data)
                        np.testing.assert_array_equal(data, output)
