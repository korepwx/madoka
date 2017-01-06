# -*- coding: utf-8 -*-
import unittest

import numpy as np
import tensorflow as tf

from madoka.utils import tfhelper, assert_raises_message
from madoka.utils.tfcompat import global_variables_initializer


class FunctionTestCase(unittest.TestCase):
    """Unit test cases for TensorFlow function wrapper."""

    def test_make_function(self):
        """Test make function."""
        graph = tf.Graph()
        with graph.as_default():
            a = tf.placeholder(name='a', shape=(), dtype=np.int32)
            b = tf.placeholder(name='b', shape=(), dtype=np.int32)
            c = tf.placeholder(name='c', shape=(), dtype=np.int32)
            fn = tfhelper.make_function(
                inputs=[a, b],
                outputs=(a + b + c),
                givens={c: np.array(1000, dtype=np.int32)}
            )

        with tf.Session(graph=graph):
            self.assertEqual(fn(1, 2), 1003)

        # test calling function with None outputs.
        with graph.as_default():
            fn = tfhelper.make_function(outputs=[None, tf.constant(1)])

        with tf.Session(graph=graph):
            self.assertEqual(fn(), (None, 1))

    def test_args_check(self):
        """Test argument check when calling function."""

        graph = tf.Graph()
        with graph.as_default():
            a = tf.placeholder(name='a', shape=(), dtype=np.int32)
            b = tf.placeholder(name='b', shape=(), dtype=np.int32)
            c = tf.Variable(2, name='c', dtype=np.int32)

        with tf.Session(graph=graph) as sess:
            sess.run(global_variables_initializer())

            # test calling function with no argument.
            fn0 = tfhelper.make_function(outputs=c)
            self.assertEquals(fn0(), 2)
            with assert_raises_message(
                    self, ValueError,
                    'Function only accepts unnamed arguments.'):
                fn0(a=2)
            with assert_raises_message(
                    self, ValueError,
                    'Require 0 unnamed arguments, but got 1.'):
                fn0(1)

            # test calling function with one unnamed argument.
            fn1 = tfhelper.make_function(inputs=a, outputs=2 * a)
            self.assertEquals(fn1(2), 4)
            with assert_raises_message(
                    self, ValueError,
                    'Function only accepts unnamed arguments.'):
                fn1(a=2)
            with assert_raises_message(
                    self, ValueError,
                    'Require 1 unnamed arguments, but got 0.'):
                fn1()
            with assert_raises_message(
                    self, ValueError,
                    'Require 1 unnamed arguments, but got 2.'):
                fn1(2, 3)

            # test calling function with two unnamed arguments.
            fn2 = tfhelper.make_function(inputs=[a, b], outputs=a + b)
            self.assertEquals(fn2(2, 3), 5)
            with assert_raises_message(
                    self, ValueError,
                    'Function only accepts unnamed arguments.'):
                fn2(a=2, b=3)
            with assert_raises_message(
                    self, ValueError,
                    'Function only accepts unnamed arguments.'):
                fn2(2, b=3)
            with assert_raises_message(
                    self, ValueError,
                    'Require 2 unnamed arguments, but got 1.'):
                fn2(2)
            with assert_raises_message(
                    self, ValueError,
                    'Require 2 unnamed arguments, but got 3.'):
                fn2(2, 3, 4)

            # test calling function with two named arguments.
            fn3 = tfhelper.make_function(inputs={'a': a, 'b': b}, outputs=a + b)
            self.assertEquals(fn3(a=3, b=4), 7)
            with assert_raises_message(
                    self, ValueError,
                    'Function only accepts named arguments.'):
                fn3(3, 4)
            with assert_raises_message(
                    self, ValueError,
                    'Function only accepts named arguments.'):
                fn3(3, b=4)
            with assert_raises_message(
                    self, ValueError,
                    'Named argument b is required but not specified.'):
                fn3(a=3)
            with assert_raises_message(
                    self, ValueError,
                    'Unexpected named argument c.'):
                fn3(a=3, b=4, c=5)

    def test_updates(self):
        """Test variable updates by function."""

        graph = tf.Graph()
        with graph.as_default():
            a = tf.placeholder(name='a', shape=(), dtype=np.int32)
            b = tf.placeholder(name='b', shape=(), dtype=np.int32)
            c = tf.Variable(2, name='c', dtype=np.int32)
            d = tf.Variable(3, name='d', dtype=np.int32)

        with tf.Session(graph=graph) as sess:
            updates = tf.assign(c, a + b)
            fn = tfhelper.make_function(inputs=[a, b],
                                        outputs=[tf.constant(1), None],
                                        updates=updates)
            sess.run(global_variables_initializer())
            self.assertEquals(fn(2, 3), (1, None))
            self.assertEquals(tfhelper.get_variable_values(c), 5)

        with tf.Session(graph=graph) as sess:
            updates = [tf.assign(c, a + b), [tf.assign(d, a * b)]]
            fn = tfhelper.make_function(inputs=[a, b], updates=updates)
            sess.run(global_variables_initializer())
            fn(3, 7)
            self.assertEquals(tfhelper.get_variable_values([c, d]), (10, 21),
                              msg='Should merge updates.')
