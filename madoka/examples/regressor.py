# -*- coding: utf-8 -*-
"""Curve-fitting example."""

import math

import click
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

import madoka
from madoka import config
from madoka.nn import LinearRegression
from madoka.report import regression_report
from madoka.train import TrainStorage, ScalingMaxSteps, LossTrainer
from madoka.utils import tfhelper, split_numpy_array
from madoka.utils.tfcompat import global_variables_initializer

madoka.utils.init_logging()


def make_data(n_samples, x_xrange=math.pi * 2):
    X = np.random.random_sample(size=[n_samples]) * x_xrange
    y = np.sin(X)
    noise = np.random.normal(scale=0.1, size=[n_samples])
    features = np.stack([X, X**2, X**3, X**4], axis=1)
    targets = y + noise
    return features.astype(config.floatX), targets.astype(config.floatX)

(train_X, train_y), (test_X, test_y) = \
    split_numpy_array(make_data(2000), right_portion=0.2)


def run(store):
    # Standardize every dimension of the data.
    # this is very important for a regression task!
    try:
        prep = store.load_object('scaler.pkl.gz')
        train_X_scaled = prep.transform(train_X)
    except IOError:
        prep = StandardScaler()
        train_X_scaled = prep.fit_transform(train_X)
        store.save_object(prep, 'scaler.pkl.gz')

    # now start to fit the linear regression model.
    graph = tf.Graph()
    with graph.as_default():
        input_ph = tfhelper.make_placeholder_for('input', train_X)
        label_ph = tfhelper.make_placeholder_for('label', train_y)

        # compose the model
        lr = LinearRegression(input_ph, 1)
        loss = lr.get_loss(label_ph)
        trainer = (LossTrainer(max_steps=ScalingMaxSteps(1000),
                               summary_dir=store.summary_dir,
                               checkpoint_dir=store.checkpoint_dir).
                   set_loss(loss, (input_ph, label_ph)).
                   set_data(train_X_scaled, train_y)
                   )
        reg = tfhelper.Regressor(input_ph, lr.output)

        with tf.Session() as sess:
            sess.run(global_variables_initializer())

            # train the model
            trainer.run()
            store.save_session()

            # evaluate the model
            test_X_scaled = prep.transform(test_X)
            y_pred = reg.predict(test_X_scaled)
            report = regression_report(test_y, y_pred, name='Curve Fitting')
            store.save_report(report)


@click.option('-R', '--recover', help='Recover training.', is_flag=True)
@click.option('--recover-from', help='Training directory to recover from.',
              default='latest')
@click.command()
def main(recover, recover_from):
    train_root = '/tmp/curve-train'
    if recover:
        store = TrainStorage.open(train_root, recover_from)
    else:
        store = TrainStorage.create(train_root)
    with store:
        run(store)


if __name__ == '__main__':
    main()
