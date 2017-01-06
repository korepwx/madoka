# -*- coding: utf-8 -*-
"""MNIST digit handwriting recognition example."""

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

import madoka
from madoka.nn import LogisticRegression, dropout
from madoka.report import classification_report
from madoka.train import TrainStorage, ScalingMaxSteps, LossTrainer
from madoka.utils import datasets, tfhelper

madoka.utils.init_logging()


def build_mlp(net):
    """Build a multi-layer perceptron."""
    net = slim.fully_connected(net, 128, activation_fn=tf.nn.relu)
    net = slim.fully_connected(net, 32, activation_fn=tf.nn.relu)
    net = dropout(net, keep_prob=0.5)
    return net

(train_X, train_y), (test_X, test_y) = \
    datasets.load_mnist(flatten_images=True, label_dtype=np.int32)

graph = tf.Graph()
with graph.as_default():
    input_ph = tfhelper.make_placeholder_for('input', train_X)
    label_ph = tfhelper.make_placeholder_for('label', train_y)

    # we will demonstrate how to recover from a previous running session
    with TrainStorage.create('/tmp/mnist-train', max_versions=3) as store:
        # compose the model
        net = build_mlp(input_ph)
        lr = LogisticRegression(net, 10)
        loss = lr.get_loss(label_ph)
        trainer = (LossTrainer(max_steps=ScalingMaxSteps(1000),
                               summary_dir=store.summary_dir,
                               checkpoint_dir=store.checkpoint_dir).
                   set_loss(loss, (input_ph, label_ph)).
                   set_data(train_X, train_y)
                   )
        clf = tfhelper.Classifier(input_ph, lr.output, lr.label, lr.log_proba)

        # now restore from the previous session
        with tf.Session() as sess:
            trainer.run()

            # finally, save the model
            store.save_session()

            # evaluate the model
            y_pred = clf.predict(test_X)
            y_proba = clf.predict_proba(test_X)
            report = classification_report(
                test_y, y_pred, y_proba, name='MNIST')
            store.save_report(report)

            np.testing.assert_array_equal(
                np.argmax(y_proba, 1),
                clf.predict(test_X)
            )
            np.testing.assert_array_equal(
                np.argmax(clf.predict_log_proba(test_X), 1),
                clf.predict(test_X)
            )
