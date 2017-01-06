# -*- coding: utf-8 -*-
import unittest

import numpy as np
import tensorflow as tf

from madoka.dataflow import DataFlow
from madoka.utils.tfhelper import (batch_collect_predict, BasePredictor,
                                   Classifier, Regressor)


class WrapperTestCase(unittest.TestCase):
    """Unit tests for wrappers."""

    BATCH_COLLECT_FLOW = DataFlow.from_numpy([
        np.arange(16).reshape((8, 2)),
        np.arange(16, 32).reshape((8, 2))
    ])

    def _check_with(self, f1, f2, res1, res2, mode):
        flow = self.BATCH_COLLECT_FLOW

        # test with null batch size
        np.testing.assert_almost_equal(
            res1, batch_collect_predict(f1, flow, mode=mode))
        np.testing.assert_almost_equal(
            res2, batch_collect_predict(f2, flow, mode=mode))

        # test with full batch size
        np.testing.assert_almost_equal(
            res1, batch_collect_predict(f1, flow, 8, mode=mode))
        np.testing.assert_almost_equal(
            res2, batch_collect_predict(f2, flow, 8, mode=mode))

        # test with divisible batch size
        np.testing.assert_almost_equal(
            res1, batch_collect_predict(f1, flow, 4, mode=mode))
        np.testing.assert_almost_equal(
            res2, batch_collect_predict(f2, flow, 4, mode=mode))

        # test with indivisible batch size
        np.testing.assert_almost_equal(
            res1, batch_collect_predict(f1, flow, 5, mode=mode))
        np.testing.assert_almost_equal(
            res2, batch_collect_predict(f2, flow, 5, mode=mode))

    def test_batch_collect_predict_concat(self):
        """Test function `batch_collect_predict()` in concat mode."""
        def f1(x, y):
            return x + y

        def f2(x, y):
            return (x + y, x * y)

        flow = self.BATCH_COLLECT_FLOW
        res1 = f1(*flow.all())
        res2 = f2(*flow.all())
        self._check_with(f1, f2, res1, res2, mode='concat')

    def test_batch_collect_predict_sum(self):
        """Test function `batch_collect_predict()` in sum mode."""
        def f1(x, y):
            return x + y

        def f2(x, y):
            return (x + y), (x * y)

        flow = self.BATCH_COLLECT_FLOW
        res1 = np.sum(f1(*flow.all()), axis=0)
        res2 = [np.sum(v, axis=0) for v in f2(*flow.all())]
        self._check_with(f1, f2, res1, res2, mode='sum')

    def test_batch_collect_predict_average(self):
        """Test function `batch_collect_predict()` in average mode."""
        def f1(x, y):
            return x + y

        def f2(x, y):
            return (x + y), (x * y)

        flow = self.BATCH_COLLECT_FLOW
        res1 = np.average(f1(*flow.all()), axis=0)
        res2 = [np.average(v, axis=0) for v in f2(*flow.all())]
        self._check_with(f1, f2, res1, res2, mode='average')

    def test_BasePredictor(self):
        """Test class `BasePredictor`."""
        X = np.arange(16, dtype=np.int32)
        Y = X + 16
        Z = Y + 16
        sum_XYZ = X + Y + Z
        sum_XYZ_2d = sum_XYZ.reshape([-1, 1])
        flow_XYZ = DataFlow.from_numpy([X, Y, Z])

        with tf.Graph().as_default():
            x = tf.placeholder(name='x', shape=(None,), dtype=tf.int32)
            y = tf.placeholder(name='y', shape=(None,), dtype=tf.int32)
            z = tf.placeholder(name='z', shape=(None,), dtype=tf.int32)
            sum_xyz = x + y + z
            sum_xyz_2d = tf.reshape(sum_xyz, [-1, 1])

            # build predictors
            pred_xyz = BasePredictor((x, y, z), sum_xyz)
            pred_xyz_2d = BasePredictor((x, y, z), sum_xyz_2d)
            pred_xyz_b = BasePredictor((x, y, z), sum_xyz, 5)
            pred_xyz_2d_b = BasePredictor((x, y, z), sum_xyz_2d, 5)

            with tf.Session():
                # test to use numpy arrays only
                np.testing.assert_array_equal(
                    pred_xyz.predict(X, Y, Z), sum_XYZ)
                np.testing.assert_array_equal(
                    pred_xyz_2d.predict(X, Y, Z), sum_XYZ_2d)

                # test to use numpy arrays only, in batches
                np.testing.assert_array_equal(
                    pred_xyz_b.predict(X, Y, Z), sum_XYZ)
                np.testing.assert_array_equal(
                    pred_xyz_2d_b.predict(X, Y, Z), sum_XYZ_2d)

                # test to use one data flow only
                np.testing.assert_array_equal(
                    pred_xyz.predict(flow_XYZ), sum_XYZ)
                np.testing.assert_array_equal(
                    pred_xyz_2d.predict(flow_XYZ), sum_XYZ_2d)

                # test to use one data flow only, in batches
                np.testing.assert_array_equal(
                    pred_xyz_b.predict(flow_XYZ), sum_XYZ)
                np.testing.assert_array_equal(
                    pred_xyz_2d_b.predict(flow_XYZ), sum_XYZ_2d)

                # test to use mix arrays and data flow
                args = [X, DataFlow.from_numpy(Y), Z]
                np.testing.assert_array_equal(
                    pred_xyz.predict(*args), sum_XYZ)
                np.testing.assert_array_equal(
                    pred_xyz_2d.predict(*args), sum_XYZ_2d)

                # test to use mix arrays and data flow, in batches
                args = [DataFlow.from_numpy(X), Y, DataFlow.from_numpy(Z)]
                np.testing.assert_array_equal(
                    pred_xyz_b.predict(*args), sum_XYZ)
                np.testing.assert_array_equal(
                    pred_xyz_2d_b.predict(*args), sum_XYZ_2d)

    def test_Classifier(self):
        """Test class `Classifier`."""
        # test binary predictions
        X = np.asarray([0.1, 0.5, 0.4, 0.7, 0.2], dtype=np.float32)
        X_2d = X.reshape([-1, 1])
        X_log = np.log(X)
        X_2d_log = np.log(X_2d)
        Y = (X >= 0.5).astype(np.int32)

        with tf.Graph().as_default():
            x = tf.placeholder(name='x', dtype=tf.float32, shape=(None,))
            x_2d = tf.placeholder(name='x2', dtype=tf.float32, shape=(None, 1))

            with tf.Session():
                clf1 = Classifier(x, x)
                np.testing.assert_array_equal(
                    clf1.predict(X), Y)
                np.testing.assert_almost_equal(
                    clf1.predict_proba(X), X)
                np.testing.assert_almost_equal(
                    clf1.predict_log_proba(X), X_log)

                clf2 = Classifier(x_2d, x_2d)
                np.testing.assert_array_equal(
                    clf2.predict(X_2d), Y)
                np.testing.assert_almost_equal(
                    clf2.predict_proba(X_2d), X_2d)
                np.testing.assert_almost_equal(
                    clf2.predict_log_proba(X_2d), X_2d_log)

        # test multi-class predictions
        X = np.asarray([[0.1, 0.5, 0.4], [0.2, 0.2, 0.6], [0.9, 0.05, 0.05]],
                       dtype=np.float32)
        X_log = np.log(X)
        Y = np.argmax(X, axis=1)

        with tf.Graph().as_default():
            x = tf.placeholder(name='x', dtype=tf.float32, shape=(None, 3))

            with tf.Session():
                clf = Classifier(x, x)
                np.testing.assert_array_equal(clf.predict(X), Y)
                np.testing.assert_almost_equal(clf.predict_proba(X), X)
                np.testing.assert_almost_equal(clf.predict_log_proba(X), X_log)

    def test_Regressor(self):
        """Test class `Regressor`."""
        # test 1-d regression
        X = np.arange(16, dtype=np.int32)
        X_2d = X.reshape([-1, 1])

        with tf.Graph().as_default():
            x = tf.placeholder(name='x', dtype=tf.float32, shape=(None,))
            x_2d = tf.placeholder(name='x2', dtype=tf.float32, shape=(None, 1))

            with tf.Session():
                reg1 = Regressor(x, x)
                np.testing.assert_array_equal(reg1.predict(X), X)
                reg2 = Regressor(x_2d, x_2d)
                np.testing.assert_array_equal(reg2.predict(X_2d), X)

        # test 2-d regression
        X = np.arange(16, dtype=np.int32).reshape([8, 2])

        with tf.Graph().as_default():
            x = tf.placeholder(name='x', dtype=tf.float32, shape=(None, 2))
            with tf.Session():
                reg = Regressor(x, x)
                np.testing.assert_array_equal(reg.predict(X), X)

if __name__ == '__main__':
    unittest.main()
