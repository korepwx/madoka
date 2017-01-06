# -*- coding: utf-8 -*-
"""MNIST digit handwriting recognition example."""

from logging import getLogger

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

import madoka
from madoka.nn import LogisticRegression
from madoka.train import (TrainStorage, AdaboostClassifierTrainer,
                          WeightedLossTrainer)
from madoka.utils import datasets, tfhelper
from madoka.utils.tfhelper import variable_space

madoka.utils.init_logging()


def build_model(input_ph, label_ph, name):
    from madoka.utils.tfhelper import VariableSelector as vs
    with variable_space(name) as s:
        lr = LogisticRegression(input_ph, 10)
        loss = lr.get_loss(label_ph)
        t = WeightedLossTrainer(
            max_steps=3000,
            early_stopping=True,
            trainable_vars=vs.scope(s.name) & vs.trainable(),
            restorable_vars=vs.scope(s.name) | vs.training_states() |
                            vs.trainer_slots(),
            checkpoint_steps=1000
        )
        t.set_loss(loss)
        return lr, t

(train_X, train_y), (test_X, test_y) = \
    datasets.load_mnist(flatten_images=True, label_dtype=np.int32)

graph = tf.Graph()
with graph.as_default():
    input_ph = tfhelper.make_placeholder_for('input', train_X)
    label_ph = tfhelper.make_placeholder_for('label', train_y)

    # we will demonstrate how to recover from a previous running session
    with TrainStorage.create('/tmp/adaboost-train', max_versions=3) as store:
        # compose the models
        outputs = []
        trainer = AdaboostClassifierTrainer(summary_dir=store.summary_dir,
                                            checkpoint_dir=store.checkpoint_dir)
        trainer.set_placeholders((input_ph, label_ph))
        trainer.set_data(train_X, train_y)
        for i in range(5):
            l, t = build_model(input_ph, label_ph, 'NeuralNetwork%d' % i)
            trainer.add(t, l.output)
            outputs.append(l.output)

        # now restore from the previous session
        with tf.Session() as sess:
            trainer.run()

            # get the output tensor
            clf = trainer.ensemble_classifier(input_ph)

            # finally, save the model
            store.save_session()

            # evaluate the model
            y_pred = clf.predict(test_X)
            getLogger(__name__).info(
                'Classification Report:\n%s',
                classification_report(test_y, y_pred, digits=5)
            )

            np.testing.assert_array_equal(
                np.argmax(clf.predict_proba(test_X), 1),
                clf.predict(test_X)
            )
            np.testing.assert_array_equal(
                np.argmax(clf.predict_log_proba(test_X), 1),
                clf.predict(test_X)
            )
