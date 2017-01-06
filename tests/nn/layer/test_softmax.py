# -*- coding: utf-8 -*-
import unittest

import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression

from madoka import config, nn
from madoka.utils import tfhelper
from madoka.utils.tfcompat import global_variables_initializer


class LogisticRegressionUnitTest(unittest.TestCase):

    def _do_test_predicting(self, target_num, lr, W, b, X, y):
        lr.fit(X, y)
        lr.coef_ = W.T
        lr.intercept_ = b
        np.testing.assert_array_equal(lr.predict(X), y)

        graph = tf.Graph()
        with graph.as_default():
            input_ph = tf.placeholder(name='inputs', shape=(None, W.shape[0]),
                                      dtype=config.floatX)
            lr2 = nn.LogisticRegression(input_ph, target_num, W=W, b=b)
            predict_fn = tfhelper.make_function(
                inputs=input_ph,
                outputs=[lr2.label, lr2.proba, lr2.log_proba]
            )

        def assert_almost_equal(a, b, rtol=1e-5, mean_err=1e-6):
            np.testing.assert_allclose(a, b, rtol=rtol)
            self.assertLess(np.mean(np.abs(a - b)), mean_err)

        with tf.Session(graph=graph) as sess:
            sess.run(global_variables_initializer())
            predict, proba, log_proba = predict_fn(X)
            np.testing.assert_array_equal(predict, y)
            assert_almost_equal(proba, lr.predict_proba(X))
            assert_almost_equal(log_proba, lr.predict_log_proba(X))

    def test_binary_predicting(self):
        """Test binary softmax classifier."""
        # When target_num == 2, LogisticRegression from scikit-learn uses
        # sigmoid, so does our LogisticRegression implementation.
        W = np.asarray([4, -1, 2, 3], dtype=np.float32).reshape([-1, 1])
        b = np.asarray([1], dtype=np.float32)
        X = np.asarray([[1, 1, -1, -0.1], [-0.5, 1, 1, -0.5]], dtype=np.float32)
        y = np.asarray([1, 0], dtype=np.int32)
        self._do_test_predicting(2, LogisticRegression(), W, b, X, y)

    def test_categorical_predicting(self):
        """Test categorical softmax classifier."""
        W = np.asarray([[4, -1, 2, 3], [0, 1, -2, 3], [-3, 0, 1, -2]],
                       dtype=np.float32)
        b = np.asarray([1, -2, 3, 0], dtype=np.float32)
        X = np.asarray([[1, 2, -1], [-0.5, 1, 1], [0.5, 0, -1], [-2, 1, 0]],
                       dtype=np.float32)
        y = np.asarray([3, 2, 0, 1], dtype=np.int32)
        lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        self._do_test_predicting(4, lr, W, b, X, y)
