# -*- coding: utf-8 -*-
import unittest
from logging import getLogger

import sklearn.ensemble
import numpy as np
import tensorflow as tf

from madoka.dataflow import DataFlow
from madoka.train import AdaboostClassifierTrainer, Trainer
from madoka.utils.tfhelper import (variable_space,
                                   get_variable_values,
                                   ensure_variables_initialized)
from ..helper import make_multimodal_tf_splitter, MultimodalSplitter


class _TrainLogger(object):

    def __init__(self):
        self.model_weight = None
        self.trained_model = None
        self.predict_proba = None
        self.data_weights = []

    def set_trained_model(self, trained_model):
        self.trained_model = trained_model

    def set_model_weight(self, model_weight):
        self.model_weight = model_weight

    def set_predict_proba(self, predict_proba):
        self.predict_proba = predict_proba

    def add_data_weight(self, sample_weight):
        self.data_weights.append(sample_weight)

    @property
    def data_weights_normalized(self):
        w = np.asarray(self.data_weights)
        return w / w.shape[1]


class _DummyTrainer(Trainer):

    def __init__(self, train_logger):
        super(_DummyTrainer, self).__init__(max_epoch=1000)
        self.train_logger = train_logger

    def set_weight(self, weight):
        weight = get_variable_values(weight)
        self.train_logger.add_data_weight(list(weight))
        return self

    def _run(self, checkpoint_dir, train_flow, valid_flow):
        ensure_variables_initialized()
        return self


class BoostingTestCase(unittest.TestCase):
    """Unit tests for boosting trainers."""

    def make_binary_classify_data(self):
        X = [-1.4746392980019123, -1.0587966152733623, -1.002698071476765,
             -0.90663697009056821, -0.83994811136771097, -0.34937160881636065,
             -0.23990752114608749, 0.051754220565130349, 0.12137405037148863,
             0.18637432976300983, 0.24671466427154953, 0.2594707099416973,
             0.26598127933625326, 0.34296057762655263, 0.40575948664041506,
             0.69571059215037789, 0.73548866690874171, 1.3772634632150056,
             2.2166351863737797, 2.505170497912256]
        y = [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1]
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)

    def make_binary_classify_perfect_data(self):
        X = np.linspace(-1.0, 1.0, 10)
        y = (X >= 0.).astype(np.int32)
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)

    def make_multimodal_classify_data(self):
        X = [-4.160417632310514, -3.0219682482694075, -2.9315738268760283,
             -2.5238575018193403, -2.188345038138285, -1.5356707203405775,
             -1.3546776870416042, -1.2518560174576847, -1.1998524118161704,
             -1.1227311674491318, -1.0837060713856219, -0.91523042421944567,
             -0.36736902896121687, -0.36078882742019269, -0.23747991023482937,
             -0.19613483910501531, -0.18479086898824137, -0.067940163006002349,
             0.2444323642673987, 0.56900436988011993, 1.2421267837411381,
             1.2814211229794965, 1.4387267819655976, 1.4896830980037845,
             1.5062618170725237, 1.7441828221803104, 2.7207279043191175,
             2.7677784087800612, 3.1751809626760945, 3.3254862818432027]
        y = [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 2, 1, 1, 0, 1, 1, 1, 2, 2,
             2, 2, 1, 2, 2, 2, 2, 2]
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)

    def test_AdaboostClassifier(self):
        """Test Adaboost classifier trainer."""
        
        def sklearn_train(X, y, clf, algorithm='SAMME.R', n_estimators=3):
            X = X.reshape([-1, 1])
            adaboost = sklearn.ensemble.AdaBoostClassifier(
                clf,
                algorithm=algorithm,
                n_estimators=n_estimators
            )
            adaboost.fit(X, y)
            if algorithm == 'SAMME':
                # AdaBoost classifier of Scikit-Learn uses a very strange
                # way to compose the prediction probability when using
                # "SAMME" algorithm, so we instead generate our desired
                # answers here.
                def predict_proba(this, X_):
                    n_classes = this.n_classes_
                    classes = this.classes_.reshape([1, -1])
                    X_ = this._validate_X_predict(X_)

                    if this.algorithm == 'SAMME.R':
                        raise RuntimeError()
                    else:  # self.algorithm == "SAMME"
                        proba = sum(
                            w * (estimator.predict(X_).reshape([-1, 1]) ==
                                 classes)
                            for estimator, w in zip(this.estimators_,
                                                    this.estimator_weights_)
                        )

                    proba /= this.estimator_weights_.sum()
                    proba = np.exp((1. / (n_classes - 1)) * proba)
                    normalizer = proba.sum(axis=1)[:, np.newaxis]
                    normalizer[normalizer == 0.0] = 1.0
                    proba /= normalizer

                    return proba

                from functools import partial
                adaboost.predict_proba = partial(predict_proba, adaboost)
            getLogger(__name__).debug(
                'Fitted models: %s', len(adaboost.estimators_)
            )
            getLogger(__name__).debug(
                'Model weights: %s',
                list(adaboost.estimator_weights_.astype(np.float32))
            )
            getLogger(__name__).debug(
                'Predict proba: %s',
                [list(p) for p in adaboost.predict_proba(X).astype(np.float32)]
            )

        def madoka_train(X, y, train_logger, make_proba, algorithm='SAMME.R',
                         predict_batch_size=None, n_estimators=3):
            with tf.Graph().as_default():
                trainer = AdaboostClassifierTrainer(
                    algorithm=algorithm,
                    predict_batch_size=predict_batch_size
                )
                input_ph = tf.placeholder(tf.float32, shape=(None,), name='X')
                label_ph = tf.placeholder(tf.int32, shape=(None,), name='y')
                proba = make_proba(input_ph)

                for i in range(n_estimators):
                    with variable_space('child%d' % i):
                        trainer.add(_DummyTrainer(train_logger), proba)
                trainer.set_placeholders((input_ph, label_ph))
                trainer.set_data_flow(DataFlow.from_numpy([X, y]),
                                      shuffle=False)

                with tf.Session():
                    try:
                        trainer.run()
                    except Exception as ex:
                        getLogger(__name__).warn('failed.', exc_info=1)
                        raise ex
                    train_logger.set_model_weight(trainer.model_weight)
                    train_logger.set_trained_model(trainer.trained_model_num)

                    clf = trainer.ensemble_classifier(input_ph)
                    train_logger.set_predict_proba(clf.predict_proba(X))

        def make_binary_1d_proba(input_ph):
            return 1. / (1 + tf.exp(-input_ph))

        def make_binary_2d_proba(input_ph):
            proba = tf.reshape(1. / (1 + tf.exp(-input_ph)), [-1, 1])
            proba = tf.concat(1, [1. - proba, proba])
            return proba

        def make_multimodal_proba(input_ph):
            return make_multimodal_tf_splitter([-1.0, 1.0], input_ph)

        ###########################
        # SAMME.R algorithm tests #
        ###########################
        # these weights are generated by sklearn.ensemble.AdaboostClassifier
        SAMME_R_BINARY_MODEL_WEIGHT = [1.0, 1.0, 1.0]
        SAMME_R_BINARY_DATA_WEIGHTS = [
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
             0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            [0.028994491, 0.03569559, 0.10006046, 0.038517281, 0.039823271,
             0.050893724, 0.053756885, 0.059059672, 0.057039183, 0.066527322,
             0.053574231, 0.069003746, 0.069228739, 0.05105713, 0.074240156,
             0.042801373, 0.041958503, 0.030441104, 0.020007512, 0.017319631],
            [0.014593636, 0.022118807, 0.17380309, 0.025753947, 0.027530013,
             0.044963595, 0.050164994, 0.060550068, 0.056477975, 0.076830313,
             0.04982467, 0.082656667, 0.083196558, 0.045252789, 0.095677592,
             0.031801526, 0.03056135, 0.016086195, 0.0069489423, 0.0052072671]
        ]
        SAMME_R_BINARY_PREDICT_PROBA = [
            [0.81376153, 0.18623848], [0.74246055, 0.25753948],
            [0.73158872, 0.26841125], [0.71231151, 0.28768852],
            [0.69845426, 0.30154571], [0.58646518, 0.41353476],
            [0.55969077, 0.44030914], [0.48706433, 0.5129357],
            [0.46969366, 0.53030634], [0.4535408, 0.5464592],
            [0.43863231, 0.56136763], [0.43549386, 0.56450617],
            [0.43389401, 0.56610602], [0.41509053, 0.5849095],
            [0.3999294, 0.6000706], [0.33276391, 0.66723609],
            [0.32399142, 0.67600852], [0.20144886, 0.79855114],
            [0.098266542, 0.90173346], [0.075496554, 0.92450345]
        ]
        SAMME_R_MULTIMODAL_MODEL_WEIGHT = [1.0, 1.0, 1.0]
        SAMME_R_MULTIMODAL_DATA_WEIGHTS = [
            [0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.033333335,
             0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.033333335,
             0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.033333335,
             0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.033333335,
             0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.033333335,
             0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.033333335],
            [3.5041964e-10, 1.5645844e-08, 2.1195792e-08, 8.5922323e-08,
             3.1245048e-07, 0.0021352645, 3.5613208e-05, 0.00084949296,
             9.8551704e-05, 0.00016410233, 0.00021250022, 0.00027990813,
             4.9469389e-05, 0.9181059, 3.4721306e-05, 3.1324478e-05,
             0.075737923, 2.3646204e-05, 3.5345867e-05, 9.1142407e-05,
             7.4568903e-05, 5.7574118e-05, 2.0608069e-05, 1.4834612e-05,
             0.0019439214, 3.0467497e-06, 4.3303544e-08, 3.6880593e-08,
             9.3685015e-09, 5.6709095e-09],
            [1.4469086e-19, 2.8844447e-16, 5.293751e-16, 8.6991385e-15,
             1.1503412e-13, 5.3723907e-06, 1.4944673e-09, 8.5032286e-07,
             1.1444387e-08, 3.1731727e-08, 5.3208748e-08, 9.2319894e-08,
             2.8836153e-09, 0.99322999, 1.4205493e-09, 1.1561972e-09,
             0.0067591341, 6.5885086e-10, 1.4721143e-09, 9.7882538e-09,
             6.5520882e-09, 3.9058792e-09, 5.0042487e-10, 2.5930841e-10,
             4.4526819e-06, 1.0937997e-11, 2.2095873e-15, 1.6027288e-15,
             1.0342e-16, 3.7893931e-17]
        ]
        SAMME_R_MULTIMODAL_PREDICT_PROBA = [
            [1.0, 9.2422331e-10, 4.1959671e-14],
            [0.9999997, 2.7573543e-07, 1.2518373e-11],
            [0.99999958, 4.3477888e-07, 1.973894e-11],
            [0.99999642, 3.5485507e-06, 1.6110449e-10],
            [0.99997538, 2.4606599e-05, 1.1171576e-09],
            [0.99484968, 0.0051500876, 2.339217e-07],
            [0.97094136, 0.02905732, 1.3207099e-06],
            [0.9239288, 0.076067753, 3.4600771e-06],
            [0.87898147, 0.12101303, 5.5076134e-06],
            [0.77180624, 0.22818339, 1.0397297e-05],
            [0.69661927, 0.3033669, 1.3833903e-05],
            [0.30113488, 0.69883305, 3.2053551e-05],
            [0.0020685967, 0.99787891, 5.2521027e-05],
            [0.0019460187, 0.99800122, 5.2768894e-05],
            [0.00063630549, 0.99930447, 5.9205475e-05],
            [0.00044351316, 0.99949408, 6.2394924e-05],
            [0.00040227157, 0.99953431, 6.3390602e-05],
            [0.00015328053, 0.99976903, 7.7703466e-05],
            [5.8730377e-05, 0.99926466, 0.00067664043],
            [4.7364025e-05, 0.9859482, 0.014004409],
            [3.7804245e-06, 0.083102547, 0.91689366],
            [2.6346736e-06, 0.057937026, 0.94206035],
            [5.9085829e-07, 0.013004749, 0.98699468],
            [3.6266766e-07, 0.007983637, 0.99201602],
            [3.0954666e-07, 0.0068145664, 0.9931851],
            [3.3994962e-08, 0.00074866746, 0.99925131],
            [5.7641381e-11, 1.2696343e-06, 0.99999875],
            [4.5304996e-11, 9.9790793e-07, 0.99999899],
            [5.8003559e-12, 1.2776133e-07, 0.99999988],
            [2.7316717e-12, 6.0169072e-08, 0.99999994]
        ]

        # test SAMME.R on binary classifier with early termination
        X, y = self.make_binary_classify_perfect_data()
        train_logger = _TrainLogger()
        madoka_train(X, y, train_logger, make_binary_1d_proba, 'SAMME.R')
        self.assertEquals(train_logger.trained_model, 1)
        np.testing.assert_almost_equal(train_logger.model_weight, [1., 0., 0.])
        np.testing.assert_almost_equal(
            train_logger.data_weights_normalized,
            [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
        )

        # binary classifier data
        X, y = self.make_binary_classify_data()

        # test SAMME.R on binary classifier with 1d proba
        train_logger = _TrainLogger()
        madoka_train(X, y, train_logger, make_binary_1d_proba, 'SAMME.R')
        self.assertEquals(train_logger.trained_model, 3)
        np.testing.assert_almost_equal(
            train_logger.model_weight, SAMME_R_BINARY_MODEL_WEIGHT)
        np.testing.assert_almost_equal(
            train_logger.data_weights_normalized, SAMME_R_BINARY_DATA_WEIGHTS)
        np.testing.assert_almost_equal(
            train_logger.predict_proba, SAMME_R_BINARY_PREDICT_PROBA)

        # test SAMME.R on binary classifier with 2d proba
        train_logger = _TrainLogger()
        madoka_train(X, y, train_logger, make_binary_2d_proba, 'SAMME.R',
                     predict_batch_size=3)
        self.assertEquals(train_logger.trained_model, 3)
        np.testing.assert_almost_equal(
            train_logger.model_weight, SAMME_R_BINARY_MODEL_WEIGHT)
        np.testing.assert_almost_equal(
            train_logger.data_weights_normalized, SAMME_R_BINARY_DATA_WEIGHTS)
        np.testing.assert_almost_equal(
            train_logger.predict_proba, SAMME_R_BINARY_PREDICT_PROBA)

        # multimodal classifier data
        X, y = self.make_multimodal_classify_data()

        # test SAMME.R on multimodal classifier with 2d proba
        train_logger = _TrainLogger()
        madoka_train(X, y, train_logger, make_multimodal_proba, 'SAMME.R')
        self.assertEquals(train_logger.trained_model, 3)
        np.testing.assert_almost_equal(
            train_logger.model_weight, SAMME_R_MULTIMODAL_MODEL_WEIGHT)
        np.testing.assert_almost_equal(
            train_logger.data_weights_normalized,
            SAMME_R_MULTIMODAL_DATA_WEIGHTS
        )
        np.testing.assert_almost_equal(
            train_logger.predict_proba, SAMME_R_MULTIMODAL_PREDICT_PROBA)

        #########################
        # SAMME algorithm tests #
        #########################
        SAMME_BINARY_MODEL_WEIGHT = [1.0986123, 0.0, 0.0]
        SAMME_BINARY_DATA_WEIGHTS = [
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
             0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            [0.033333335, 0.033333335, 0.1, 0.033333335, 0.033333335,
             0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.1,
             0.033333335, 0.1, 0.1, 0.033333335, 0.1, 0.033333335, 0.033333335,
             0.033333335, 0.033333335, 0.033333335]
        ]
        SAMME_BINARY_PREDICT_PROBA = [
            [0.7310586, 0.26894143], [0.7310586, 0.26894143],
            [0.7310586, 0.26894143], [0.7310586, 0.26894143],
            [0.7310586, 0.26894143], [0.7310586, 0.26894143],
            [0.7310586, 0.26894143], [0.26894143, 0.7310586],
            [0.26894143, 0.7310586], [0.26894143, 0.7310586],
            [0.26894143, 0.7310586], [0.26894143, 0.7310586],
            [0.26894143, 0.7310586], [0.26894143, 0.7310586],
            [0.26894143, 0.7310586], [0.26894143, 0.7310586],
            [0.26894143, 0.7310586], [0.26894143, 0.7310586],
            [0.26894143, 0.7310586], [0.26894143, 0.7310586]
        ]
        SAMME_MULTIMODAL_MODEL_WEIGHT = [2.3025851, 2.220446e-16, 0.0]
        SAMME_MULTIMODAL_DATA_WEIGHTS = [
            [0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.033333335,
             0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.033333335,
             0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.033333335,
             0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.033333335,
             0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.033333335,
             0.033333335, 0.033333335, 0.033333335, 0.033333335, 0.033333335],
            [0.013333334, 0.013333334, 0.013333334, 0.013333334, 0.013333334,
             0.13333334, 0.013333334, 0.13333334, 0.013333334, 0.013333334,
             0.013333334, 0.013333334, 0.013333334, 0.13333334, 0.013333334,
             0.013333334, 0.13333334, 0.013333334, 0.013333334, 0.013333334,
             0.013333334, 0.013333334, 0.013333334, 0.013333334, 0.13333334,
             0.013333334, 0.013333334, 0.013333334, 0.013333334, 0.013333334],
            [0.013333334, 0.013333334, 0.013333334, 0.013333334, 0.013333334,
             0.13333334, 0.013333334, 0.13333334, 0.013333334, 0.013333334,
             0.013333334, 0.013333334, 0.013333334, 0.13333334, 0.013333334,
             0.013333334, 0.13333334, 0.013333334, 0.013333334, 0.013333334,
             0.013333334, 0.013333334, 0.013333334, 0.013333334, 0.13333334,
             0.013333334, 0.013333334, 0.013333334, 0.013333334, 0.013333334]
        ]
        SAMME_MULTIMODAL_PREDICT_PROBA = [
            [0.45186275, 0.27406862, 0.27406862],
            [0.45186275, 0.27406862, 0.27406862],
            [0.45186275, 0.27406862, 0.27406862],
            [0.45186275, 0.27406862, 0.27406862],
            [0.45186275, 0.27406862, 0.27406862],
            [0.45186275, 0.27406862, 0.27406862],
            [0.45186275, 0.27406862, 0.27406862],
            [0.45186275, 0.27406862, 0.27406862],
            [0.45186275, 0.27406862, 0.27406862],
            [0.45186275, 0.27406862, 0.27406862],
            [0.45186275, 0.27406862, 0.27406862],
            [0.27406862, 0.45186275, 0.27406862],
            [0.27406862, 0.45186275, 0.27406862],
            [0.27406862, 0.45186275, 0.27406862],
            [0.27406862, 0.45186275, 0.27406862],
            [0.27406862, 0.45186275, 0.27406862],
            [0.27406862, 0.45186275, 0.27406862],
            [0.27406862, 0.45186275, 0.27406862],
            [0.27406862, 0.45186275, 0.27406862],
            [0.27406862, 0.45186275, 0.27406862],
            [0.27406862, 0.27406862, 0.45186275],
            [0.27406862, 0.27406862, 0.45186275],
            [0.27406862, 0.27406862, 0.45186275],
            [0.27406862, 0.27406862, 0.45186275],
            [0.27406862, 0.27406862, 0.45186275],
            [0.27406862, 0.27406862, 0.45186275],
            [0.27406862, 0.27406862, 0.45186275],
            [0.27406862, 0.27406862, 0.45186275],
            [0.27406862, 0.27406862, 0.45186275],
            [0.27406862, 0.27406862, 0.45186275]
        ]

        # test SAMME on binary classifier with early termination
        X, y = self.make_binary_classify_perfect_data()
        train_logger = _TrainLogger()
        madoka_train(X, y, train_logger, make_binary_1d_proba, 'SAMME')
        self.assertEquals(train_logger.trained_model, 1)
        np.testing.assert_almost_equal(train_logger.model_weight, [1., 0., 0.])
        np.testing.assert_almost_equal(
            train_logger.data_weights_normalized,
            [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
        )

        # binary classifier data
        X, y = self.make_binary_classify_data()

        # test SAMME on binary classifier with 1d proba
        train_logger = _TrainLogger()
        madoka_train(X, y, train_logger, make_binary_1d_proba, 'SAMME')
        self.assertEquals(train_logger.trained_model, 1)
        np.testing.assert_almost_equal(
            train_logger.model_weight, SAMME_BINARY_MODEL_WEIGHT)
        np.testing.assert_almost_equal(
            train_logger.data_weights_normalized, SAMME_BINARY_DATA_WEIGHTS)
        np.testing.assert_almost_equal(
            train_logger.predict_proba, SAMME_BINARY_PREDICT_PROBA)

        # test SAMME on binary classifier with 2d proba
        train_logger = _TrainLogger()
        madoka_train(X, y, train_logger, make_binary_2d_proba, 'SAMME',
                     predict_batch_size=3)
        self.assertEquals(train_logger.trained_model, 1)
        np.testing.assert_almost_equal(
            train_logger.model_weight, SAMME_BINARY_MODEL_WEIGHT)
        np.testing.assert_almost_equal(
            train_logger.data_weights_normalized, SAMME_BINARY_DATA_WEIGHTS)
        np.testing.assert_almost_equal(
            train_logger.predict_proba, SAMME_BINARY_PREDICT_PROBA)

        # multimodal classifier data
        X, y = self.make_multimodal_classify_data()

        # test SAMME on multimodal classifier with 2d proba
        train_logger = _TrainLogger()
        sklearn_train(X, y, MultimodalSplitter([-1.0, 1.0]), 'SAMME')
        madoka_train(X, y, train_logger, make_multimodal_proba, 'SAMME')
        self.assertTrue(train_logger.trained_model in (1, 2))
        if train_logger.trained_model == 1:
            np.testing.assert_almost_equal(
                train_logger.model_weight,
                SAMME_MULTIMODAL_MODEL_WEIGHT
            )
            np.testing.assert_almost_equal(
                train_logger.data_weights_normalized,
                SAMME_MULTIMODAL_DATA_WEIGHTS[:2]
            )
        else:
            np.testing.assert_almost_equal(
                train_logger.model_weight,
                SAMME_MULTIMODAL_MODEL_WEIGHT,
                decimal=5
            )
            np.testing.assert_almost_equal(
                train_logger.data_weights_normalized,
                SAMME_MULTIMODAL_DATA_WEIGHTS
            )
