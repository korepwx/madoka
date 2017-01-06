# -*- coding: utf-8 -*-
import unittest

import numpy as np
import tensorflow as tf

from madoka import config
from madoka.utils import tfhelper
from madoka.utils.tfcompat import global_variables_initializer
from madoka.utils.tfhelper import (ensure_default_session,
                                   make_ensemble_output,
                                   make_ensemble_classifier)


class OpTester(object):

    def __init__(self, owner, tensor_op, numpy_op=None,
                 seal_args=False, dtype=tf.as_dtype(config.floatX)):
        self.owner = owner
        self.tensor_op = tensor_op
        self.numpy_op = numpy_op
        self.seal_args = seal_args
        self.dtype = dtype
        self.test_cases = []

    def convert_to_tensor(self, x):
        return tf.convert_to_tensor(x, dtype=self.dtype)

    def add(self, *args, **kwargs):
        self.test_cases.append((args, kwargs))

    def run(self):
        graph = tf.Graph()
        with graph.as_default():
            r0, r1, r2 = [], [], []
            for i, (args, kwargs) in enumerate(self.test_cases):
                kwargs = kwargs.copy()
                if self.numpy_op is None:
                    r0.append(np.asarray(kwargs.pop('result'), dtype=np.int32))
                else:
                    if self.seal_args:
                        r0.append(self.numpy_op(args, **kwargs))
                    else:
                        r0.append(self.numpy_op(*args, **kwargs))
                tensor_args = [self.convert_to_tensor(a) for a in args]
                if self.seal_args:
                    r1.append(self.tensor_op(tensor_args, **kwargs))
                else:
                    r1.append(self.tensor_op(*tensor_args, **kwargs))
            compute_r1 = tfhelper.make_function(outputs=r1)
            with tf.Session(graph=graph) as sess:
                sess.run(global_variables_initializer())
                r1 = compute_r1()
            for a, b, (args, kwargs) in zip(r1, r0, self.test_cases):
                self.owner.assertTrue(
                    a.shape == b.shape,
                    msg='%r != %r: args=%r, kwargs=%r' % (a, b, args, kwargs)
                )
                self.owner.assertTrue(
                    np.allclose(a, b),
                    msg='%r != %r: args=%r, kwargs=%r' % (a, b, args, kwargs)
                )


class OpTestCase(unittest.TestCase):

    def test_flatten(self):
        """Test flatten."""
        x = np.arange(16, dtype=np.int32).reshape([2, 4, 2])
        t = OpTester(self, tfhelper.flatten, dtype=np.int32)
        t.add(x, result=x.reshape([16]))
        t.add(x, ndim=2, result=x.reshape([2, 8]))
        t.add(x, ndim=3, result=x.reshape([2, 4, 2]))
        t.run()

    def test_make_ensemble_output(self):
        """Tests for `make_ensemble_output()`."""
        def f(tensors, weight=None):
            o = make_ensemble_output(tensors, weight)
            s = ensure_default_session()
            return s.run(o)

        def f1(tensors, weight=None):
            tensors = tf.convert_to_tensor(tensors)
            return f(tensors, weight)

        def f2(tensors, weight):
            weight = tf.convert_to_tensor(weight, preferred_dtype=tf.float32)
            return f(tensors, weight)

        def f3(tensors, weight):
            weight = [tf.convert_to_tensor(w, preferred_dtype=tf.float32)
                      for w in weight]
            return f(tensors, weight)

        def g(values, weight=None):
            if weight is None:
                ret = np.mean(values, axis=0)
            else:
                values = np.asarray(values)
                weight = np.asarray(weight)
                shape_diff = len(values.shape) - len(weight.shape)
                weight = weight.reshape(weight.shape + ((1,) * shape_diff))
                ret = np.sum(values * weight, axis=0) / float(np.sum(weight))
            return ret

        a = np.arange(3, dtype=np.float32)
        b = np.arange(9, dtype=np.float32).reshape((3, 3))

        with tf.Graph().as_default():
            va = tf.get_variable('a', initializer=a)
            vb = tf.get_variable('b', initializer=b)

            with tf.Session() as sess:
                sess.run(global_variables_initializer())

                # test null weights
                np.testing.assert_array_equal(f([va]), g([a]))
                np.testing.assert_array_equal(f([va, va+10]), g([a, a+10]))
                np.testing.assert_array_equal(f([vb]), g([b]))
                np.testing.assert_array_equal(f([vb, vb+10]), g([b, b+10]))

                # test numeric weights
                np.testing.assert_array_equal(f([va], [2]), g([a], [2]))
                np.testing.assert_array_equal(f([va, va+10], [2, 3]),
                                              g([a, a+10], [2, 3]))
                np.testing.assert_array_equal(f([vb], [2]), g([b], [2]))
                np.testing.assert_array_equal(f([vb, vb+10], [2, 3]),
                                              g([b, b+10], [2, 3]))

                # test zero weight error
                with self.assertRaises(ValueError) as cm:
                    f([va], [0])
                self.assertEquals(str(cm.exception),
                                  'Weights sum up to 0, which is not allowed.')

                # test tensor input as a whole
                np.testing.assert_array_equal(f1([va], [2]), g([a], [2]))
                np.testing.assert_array_equal(f1([va, va+10], [2, 3]),
                                              g([a, a+10], [2, 3]))
                np.testing.assert_array_equal(f1([vb], [2]), g([b], [2]))
                np.testing.assert_array_equal(f1([vb, vb+10], [2, 3]),
                                              g([b, b+10], [2, 3]))

                # test tensor weight
                np.testing.assert_array_equal(f2([va], [2]), g([a], [2]))
                np.testing.assert_array_equal(f2([va, va+10], [2, 3]),
                                              g([a, a+10], [2, 3]))
                np.testing.assert_array_equal(f2([vb], [2]), g([b], [2]))
                np.testing.assert_array_equal(f2([vb, vb+10], [2, 3]),
                                              g([b, b+10], [2, 3]))

                # test tensor weight list
                np.testing.assert_array_equal(f3([va], [2]), g([a], [2]))
                np.testing.assert_array_equal(f3([va, va+10], [2, 3]),
                                              g([a, a+10], [2, 3]))
                np.testing.assert_array_equal(f3([vb], [2]), g([b], [2]))
                np.testing.assert_array_equal(f3([vb, vb+10], [2, 3]),
                                              g([b, b+10], [2, 3]))

    def test_make_ensemble_classifier(self):
        """Tests for `make_ensemble_classifier()`."""
        def f(tensors, weight=None):
            o = make_ensemble_classifier(tensors, weight)
            s = ensure_default_session()
            return s.run(o)

        p1 = np.asarray([[0.51, 0.39, 0.1], [0.51, 0.48, 0.01]])
        p2 = np.asarray([[0.0, 0.0, 1.0], [0.49, 0.0, 0.51]])
        p3 = np.asarray([[0.51, 0.49, 0.0], [0.49, 0.0, 0.51]])
        with tf.Graph().as_default():
            vp1 = tf.get_variable('p1', initializer=p1)
            vp2 = tf.get_variable('p2', initializer=p2)
            vp3 = tf.get_variable('p3', initializer=p3)

            with tf.Session() as sess:
                sess.run(global_variables_initializer())

                # test merge with null weight
                np.testing.assert_almost_equal(
                    f([vp1, vp2, vp3]),
                    [[2./3, 0, 1./3], [1./3, 0, 2./3]]
                )

                # test merge with weight
                np.testing.assert_almost_equal(
                    f([vp1, vp2, vp3], [3, 1, 1]),
                    [[.8, 0, .2], [.6, 0, .4]]
                )
