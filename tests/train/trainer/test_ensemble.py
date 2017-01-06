# -*- coding: utf-8 -*-
import re
import unittest

import numpy as np
import tensorflow as tf

from madoka.dataflow import DataFlow
from madoka.train import LossTrainer, Monitor
from madoka.utils import TemporaryDirectory
from madoka.utils.tfhelper import variable_space
from ..helper import (EnsembleTrainerWithLogger, EventCollector,
                      MonitorEventLogger)


class EnsembleTestCase(unittest.TestCase):
    """Unit tests for ensemble trainers."""

    def test_EnsembleTrainer_run(self):
        """Test normal run of an ensemble trainer."""
        with tf.Graph().as_default():
            c = EventCollector()
            t = EnsembleTrainerWithLogger(collector=c)
            x = tf.placeholder(tf.float32, shape=(None,), name='x')
            loss = x * x
            for i in range(3):
                t2 = LossTrainer(early_stopping=False, batch_size=2,
                                 max_steps=2, name='LossTrainer%d' % i)
                t2.set_loss(loss * (i + 1), x)
                m = MonitorEventLogger(collector=c)
                t2.add_monitors(m)
                t._add(t2)
            t.set_data_flow(DataFlow.from_numpy(np.arange(5)), shuffle=False)

            with tf.Session():
                t.run()

            c.match([
                'before_training:initial_model_id=0',
                'before_child_training:model_id=0',
                'prepare_data_flow_for_child:model_id=0',
                'before_training',
                'start_training:batch_size=2,epoch_steps=2,initial_step=0,'
                'max_steps=FixedMaxSteps(2,initial_value=2)',
                'start_epoch:epoch=0',
                'start_step:step=0',
                'end_step:loss=0.5,step=0',
                'start_step:step=1',
                'end_step:loss=6.5,step=1',
                'end_epoch:avg_loss=3.5,epoch=0',
                'end_training:has_error=False',
                'after_child_training:model_id=0',
                'before_child_training:model_id=1',
                'prepare_data_flow_for_child:model_id=1',
                'before_training',
                'start_training:batch_size=2,epoch_steps=2,initial_step=0,'
                'max_steps=FixedMaxSteps(2,initial_value=2)',
                'start_epoch:epoch=0',
                'start_step:step=0',
                'end_step:loss=1.0,step=0',
                'start_step:step=1',
                'end_step:loss=13.0,step=1',
                'end_epoch:avg_loss=7.0,epoch=0',
                'end_training:has_error=False',
                'after_child_training:model_id=1',
                'before_child_training:model_id=2',
                'prepare_data_flow_for_child:model_id=2',
                'before_training',
                'start_training:batch_size=2,epoch_steps=2,initial_step=0,'
                'max_steps=FixedMaxSteps(2,initial_value=2)',
                'start_epoch:epoch=0',
                'start_step:step=0',
                'end_step:loss=1.5,step=0',
                'start_step:step=1',
                'end_step:loss=19.5,step=1',
                'end_epoch:avg_loss=10.5,epoch=0',
                'end_training:has_error=False',
                'after_child_training:model_id=2',
                'after_training',
            ])

    def test_EnsembleTrainer_checkpoint(self):
        """Test checkpoint of an ensemble trainer."""
        class Interrupted(BaseException):
            pass

        class InterruptAfterKthModel(Monitor):

            def __init__(self, k):
                super(InterruptAfterKthModel, self).__init__()
                self.remains = self.k = k

            def end_training(self, has_error=False):
                self.remains -= 1
                if self.remains <= 0:
                    raise Interrupted()

        def do_run(collector, monitor, max_steps, checkpoint_dir,
                   checkpoint_steps=2):
            with tf.Graph().as_default():
                X = np.asarray([0, 1], dtype=np.float32)
                Y = X * 3. + 5.
                x = tf.placeholder(tf.float32, shape=(None,), name='x')
                y = tf.placeholder(tf.float32, shape=(None,), name='y')

                t = EnsembleTrainerWithLogger(
                    collector=collector,
                    checkpoint_dir=checkpoint_dir
                )
                trainable_vars = []

                for i in range(3):
                    with variable_space('model%d' % i):
                        a = tf.get_variable('a', initializer=0.,
                                            dtype=tf.float32, trainable=True)
                        b = tf.get_variable('b', initializer=0.,
                                            dtype=tf.float32, trainable=True)
                        trainable_vars.extend([a, b])
                        output = a * x + b
                        loss = (output - y) ** 2
                    t2 = LossTrainer(
                        optimizer=tf.train.AdamOptimizer(0.001 * (10**i)),
                        batch_size=1,
                        max_steps=max_steps,
                        restore_max_steps=False,
                        early_stopping=False,
                        checkpoint_steps=checkpoint_steps,
                        name='LossTrainer%d' % i
                    )
                    t2.set_loss(loss)
                    t2.add_monitors(monitor)
                    t._add(t2)
                t.set_data_flow(DataFlow.from_numpy([X, Y]), shuffle=False)
                t.set_placeholders((x, y))

                with tf.Session() as sess:
                    try:
                        t.run()
                    except Interrupted:
                        pass
                    return sess.run(trainable_vars)

        with TemporaryDirectory() as path:
            # test first run
            c = EventCollector()
            m = MonitorEventLogger(collector=c)
            first_run = do_run(c, [m, InterruptAfterKthModel(2)], 3, path)
            np.testing.assert_almost_equal(
                first_run,
                [0.0013193503, 0.0029588281, 0.013193503, 0.029587373,
                 0.0, 0.0]
            )

            f = re.compile
            c.match([
                'before_training:initial_model_id=0',
                'before_child_training:model_id=0',
                'prepare_data_flow_for_child:model_id=0',
                'before_training',
                'start_training:batch_size=1,epoch_steps=2,initial_step=0,'
                'max_steps=FixedMaxSteps(3,initial_value=3)',
                'start_epoch:epoch=0',
                'start_step:step=0',
                f(r'end_step:loss=25\.0\d*,step=0'),
                'start_step:step=1',
                f(r'end_step:loss=63\.984\d*,step=1'),
                f(r'end_epoch:avg_loss=44\.492\d*,epoch=0'),
                'start_epoch:epoch=1',
                'start_step:step=2',
                f(r'end_step:loss=24\.980\d*,step=2'),
                f(r'end_epoch:avg_loss=24\.980\d*,epoch=1'),
                'end_training:has_error=False',
                'after_child_training:model_id=0',
                'before_child_training:model_id=1',
                'prepare_data_flow_for_child:model_id=1',
                'before_training',
                'start_training:batch_size=1,epoch_steps=2,initial_step=0,'
                'max_steps=FixedMaxSteps(3,initial_value=3)',
                'start_epoch:epoch=0',
                'start_step:step=0',
                f(r'end_step:loss=25\.0\d*,step=0'),
                'start_step:step=1',
                f(r'end_step:loss=63\.840\d*,step=1'),
                f(r'end_epoch:avg_loss=44\.420\d*,epoch=0'),
                'start_epoch:epoch=1',
                'start_step:step=2',
                f(r'end_step:loss=24\.801\d*,step=2'),
                f(r'end_epoch:avg_loss=24\.801\d*,epoch=1'),
                'end_training:has_error=False',
                'end_training:has_error=True',
            ])

            # test second run with restoring
            c = EventCollector()
            m = MonitorEventLogger(collector=c)
            second_run = do_run(c, m, 3, path)
            np.testing.assert_almost_equal(
                second_run,
                [0.0013193503, 0.0029588281, 0.013193503, 0.029587373,
                 0.13193503, 0.29577133]
            )

            f = re.compile
            c.match([
                'before_training:initial_model_id=1',
                'before_child_training:model_id=1',
                'prepare_data_flow_for_child:model_id=1',
                'before_training',
                'start_training:batch_size=1,epoch_steps=2,initial_step=3,'
                'max_steps=FixedMaxSteps(3,initial_value=3)',
                'end_training:has_error=False',
                'after_child_training:model_id=1',
                'before_child_training:model_id=2',
                'prepare_data_flow_for_child:model_id=2',
                'before_training',
                'start_training:batch_size=1,epoch_steps=2,initial_step=0,'
                'max_steps=FixedMaxSteps(3,initial_value=3)',
                'start_epoch:epoch=0',
                'start_step:step=0',
                f(r'end_step:loss=25\.0\d*,step=0'),
                'start_step:step=1',
                f(r'end_step:loss=62\.41\d*,step=1'),
                f(r'end_epoch:avg_loss=43\.70\d*,epoch=0'),
                'start_epoch:epoch=1',
                'start_step:step=2',
                f(r'end_step:loss=23\.052\d*,step=2'),
                f(r'end_epoch:avg_loss=23\.052\d*,epoch=1'),
                'end_training:has_error=False',
                'after_child_training:model_id=2',
                'after_training',
            ])


if __name__ == '__main__':
    unittest.main()
