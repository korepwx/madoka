# -*- coding: utf-8 -*-
import re
import unittest

import numpy as np
import tensorflow as tf

from madoka.dataflow import DataFlow
from madoka.train import LossTrainer, WeightedLossTrainer
from madoka.utils import TemporaryDirectory
from madoka.utils.tfcompat import global_variables_initializer
from ..helper import MonitorEventLogger


class LossTrainerUnitTest(unittest.TestCase):
    """Unit tests for loss trainer."""

    def test_loss_trainer(self):
        """Test loss trainer."""
        # compute the loss without validation, and without shuffling.
        with tf.Graph().as_default():
            t = LossTrainer(batch_size=32, max_steps=3, early_stopping=False)
            x = tf.placeholder(tf.float32, shape=(None,), name='x')
            loss = x * x
            t.set_loss(loss, x)
            t.set_data_flow(DataFlow.from_numpy(np.arange(65)), shuffle=False)
            m = MonitorEventLogger()
            t.add_monitors(m)
            with tf.Session() as sess:
                sess.run(global_variables_initializer())
                t.run()
            m.events.match([
                'before_training',
                'start_training:batch_size=32,epoch_steps=2,'
                + 'initial_step=0,max_steps=FixedMaxSteps(3,initial_value=3)',
                'start_epoch:epoch=0',
                'start_step:step=0',
                'end_step:loss=325.5,step=0',
                'start_step:step=1',
                'end_step:loss=2341.5,step=1',
                'end_epoch:avg_loss=1333.5,epoch=0',
                'start_epoch:epoch=1',
                'start_step:step=2',
                'end_step:loss=325.5,step=2',
                'end_epoch:avg_loss=325.5,epoch=1',
                'end_training:has_error=False',
            ])

        # train a simple model without validation and without shuffling
        with tf.Graph().as_default():
            X = np.linspace(0, 1, 32, dtype=np.float32)
            Y = X * 3. + 5.
            x = tf.placeholder(tf.float32, shape=(None,), name='x')
            y = tf.placeholder(tf.float32, shape=(None,), name='y')
            a = tf.get_variable('a', initializer=0., dtype=tf.float32,
                                trainable=True)
            b = tf.get_variable('b', initializer=0., dtype=tf.float32,
                                trainable=True)
            output = a * x + b
            loss = (output - y) ** 2

            t = LossTrainer(optimizer=tf.train.GradientDescentOptimizer(0.1),
                            batch_size=32, max_steps=3, early_stopping=False)
            t.set_loss(loss, (x, y))
            t.set_data_flow(DataFlow.from_numpy([X, Y]), shuffle=False)
            m = MonitorEventLogger()
            t.add_monitors(m)

            with tf.Session() as sess:
                sess.run(global_variables_initializer())
                t.run()

            f = re.compile
            m.events.match([
                'before_training',
                'start_training:batch_size=32,epoch_steps=1,initial_step=0,'
                + 'max_steps=FixedMaxSteps(3,initial_value=3)',
                'start_epoch:epoch=0',
                'start_step:step=0',
                f(r'end_step:loss=43\.048\d*,step=0'),
                f(r'end_epoch:avg_loss=43\.048\d*,epoch=0'),
                'start_epoch:epoch=1',
                'start_step:step=1',
                f(r'end_step:loss=23\.974\d*,step=1'),
                f(r'end_epoch:avg_loss=23\.974\d*,epoch=1'),
                'start_epoch:epoch=2',
                'start_step:step=2',
                f(r'end_step:loss=13\.353\d*,step=2'),
                f(r'end_epoch:avg_loss=13\.353\d*,epoch=2'),
                'end_training:has_error=False',
            ])

    def test_weighted_loss_trainer(self):
        """Test weighted loss trainer."""
        # compute the loss without validation, and without shuffling.
        with tf.Graph().as_default():
            t = WeightedLossTrainer(
                batch_size=2, max_steps=3, early_stopping=False)
            x = tf.placeholder(tf.float32, shape=(None,), name='x')
            loss = x * x
            t.set_loss(loss, x)
            t.set_data_flow(DataFlow.from_numpy(np.arange(1, 5)), shuffle=False)
            t.set_weight(np.arange(1, 5) / 10.)
            m = MonitorEventLogger()
            t.add_monitors(m)
            with tf.Session() as sess:
                sess.run(global_variables_initializer())
                t.run()
            f = re.compile
            m.events.match([
                'before_training',
                'start_training:batch_size=2,epoch_steps=2,'
                + 'initial_step=0,max_steps=FixedMaxSteps(3,initial_value=3)',
                'start_epoch:epoch=0',
                'start_step:step=0',
                f(r'end_step:loss=0\.45\d*,step=0'),
                'start_step:step=1',
                f(r'end_step:loss=4\.55\d*,step=1'),
                f(r'end_epoch:avg_loss=2\.5\d*,epoch=0'),
                'start_epoch:epoch=1',
                'start_step:step=2',
                f(r'end_step:loss=0\.45\d*,step=2'),
                f(r'end_epoch:avg_loss=0\.45\d*,epoch=1'),
                'end_training:has_error=False',
            ])

    def test_loss_trainer_checkpoint(self):
        """Test checkpoint of loss trainer."""
        def do_run(monitor, max_steps, checkpoint_dir, checkpoint_steps=2):
            with tf.Graph().as_default():
                X = np.asarray([0, 1], dtype=np.float32)
                Y = X * 3. + 5.
                x = tf.placeholder(tf.float32, shape=(None,), name='x')
                y = tf.placeholder(tf.float32, shape=(None,), name='y')
                a = tf.get_variable('a', initializer=0., dtype=tf.float32,
                                    trainable=True)
                b = tf.get_variable('b', initializer=0., dtype=tf.float32,
                                    trainable=True)
                output = a * x + b
                loss = (output - y) ** 2

                t = LossTrainer(
                    optimizer=tf.train.AdamOptimizer(),
                    batch_size=1,
                    max_steps=max_steps,
                    restore_max_steps=False,
                    early_stopping=False,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_steps=checkpoint_steps,
                )
                t.set_loss(loss, (x, y))
                t.set_data_flow(DataFlow.from_numpy([X, Y]), shuffle=False)
                t.add_monitors(monitor)

                with tf.Session():
                    t.run()

        with TemporaryDirectory() as path:
            # test first run
            m = MonitorEventLogger()
            do_run(m, 3, path)
            f = re.compile
            m.events.match([
                'before_training',
                'start_training:batch_size=1,epoch_steps=2,initial_step=0,'
                + 'max_steps=FixedMaxSteps(3,initial_value=3)',
                'start_epoch:epoch=0',
                'start_step:step=0',
                f(r'end_step:loss=25\.0\d*,step=0'),
                'start_step:step=1',
                f(r'end_step:loss=63\.984\d*,step=1'),
                f(r'end_epoch:avg_loss=44\.492\d*,epoch=0'),
                'start_epoch:epoch=1',
                'start_step:step=2',
                f(r'end_step:loss=24\.9801\d*,step=2'),
                f(r'end_epoch:avg_loss=24\.9801\d*,epoch=1'),
                'end_training:has_error=False',
            ])

            # test second run with restoring
            m = MonitorEventLogger()
            do_run(m, 4, path)
            f = re.compile
            m.events.match([
                'before_training',
                'start_training:batch_size=1,epoch_steps=2,initial_step=3,'
                + 'max_steps=FixedMaxSteps(4,initial_value=4)',
                'start_epoch:epoch=0',
                'start_step:step=3',
                f(r'end_step:loss=24\.9704\d*,step=3'),
                f(r'end_epoch:avg_loss=24\.9704\d*,epoch=0'),
                'end_training:has_error=False',
            ])

        with TemporaryDirectory() as path:
            # test third run, without restoring
            m = MonitorEventLogger()
            do_run(m, 1, path)
            f = re.compile
            m.events.match([
                'before_training',
                'start_training:batch_size=1,epoch_steps=2,initial_step=0,'
                + 'max_steps=FixedMaxSteps(1,initial_value=1)',
                'start_epoch:epoch=0',
                'start_step:step=0',
                f(r'end_step:loss=(?:25\.0|24\.99)\d*,step=0'),
                f(r'end_epoch:avg_loss=(?:25\.0|24\.99)\d*,epoch=0'),
                'end_training:has_error=False',
            ])
