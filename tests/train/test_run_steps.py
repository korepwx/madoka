# -*- coding: utf-8 -*-
import unittest

import numpy as np
import six
import tensorflow as tf

from madoka.dataflow import DataFlow, get_dataflow_context
from madoka.train import run_steps
from madoka.utils.tfcompat import scalar_summary
from madoka.utils.tfhelper import make_function, ensure_default_session
from .helper import MonitorEventLogger, SummaryWriterLogger

try:
    import tflearn
except ImportError:
    tflearn = None


class RunStepsTestCase(unittest.TestCase):
    """Unit tests for run_steps."""

    def test_basic(self):
        """Test the basic functions of run_steps."""
        def check_results(monitor, summary_writer):
            monitor.events.match([
                'before_training',
                'start_training:batch_size=2,epoch_steps=2,'
                + 'initial_step=0,max_steps=FixedMaxSteps(3,initial_value=3)',
                'start_epoch:epoch=0',
                'start_step:step=0',
                'end_step:loss=0.5,step=0',
                'start_step:step=1',
                'end_step:loss=2.5,step=1',
                'end_epoch:avg_loss=1.5,epoch=0',
                'start_epoch:epoch=1',
                'start_step:step=2',
                'end_step:loss=0.5,step=2',
                'end_epoch:avg_loss=0.5,epoch=1',
                'end_training:has_error=False',
            ])
            summary_writer.events.match([
                '0:training_loss=0.5',
                '1:training_loss=2.5',
                '2:training_loss=0.5',
            ])

        # run steps with train_fn -> loss
        with tf.Graph().as_default():
            x = tf.placeholder(tf.float32, shape=(None,), name='x')
            df = DataFlow.from_numpy(np.arange(5, dtype=np.float32))
            loss = tf.reduce_mean(x)
            loss_fn = make_function(inputs=x, outputs=loss)
            m = MonitorEventLogger()
            sw = SummaryWriterLogger()
            with tf.Session():
                run_steps(loss_fn, df, batch_size=2, max_steps=3, monitors=m,
                          summary_writer=sw)
            check_results(m, sw)

        # run steps with train_fn -> (loss, summary)
        with tf.Graph().as_default():
            x = tf.placeholder(tf.float32, shape=(None,), name='x')
            df = DataFlow.from_numpy(np.arange(5, dtype=np.float32))
            loss = tf.reduce_mean(x)
            summary = scalar_summary('training_loss', loss)
            loss_fn = make_function(inputs=x, outputs=[loss, summary])
            m = MonitorEventLogger()
            sw = SummaryWriterLogger()
            with tf.Session():
                run_steps(loss_fn, df, batch_size=2, max_steps=3, monitors=m,
                          summary_writer=sw)
            check_results(m, sw)

    def test_training_flag(self):
        """Test whether the training flag is set properly by run_steps."""
        with tf.Graph().as_default():
            x = tf.placeholder(tf.float32, shape=(None,), name='x')
            df = DataFlow.from_numpy(np.arange(5, dtype=np.float32))
            if tflearn is None:
                training_flag = None
            else:
                training_flag = tflearn.get_training_mode()
            loss = tf.reduce_mean(x)
            loss_fn = make_function(inputs=x, outputs=[loss, training_flag])
            m = MonitorEventLogger()

            # wrap the loss function to check environment
            def train_fn(*args):
                self.assertTrue(get_dataflow_context().flags.is_training)
                ret, flag = loss_fn(*args)
                if training_flag:
                    self.assertTrue(flag)
                return ret

            # wrap the data flow to check environment
            def df_wrapper(method):
                @six.wraps(method)
                def inner(*args, **kwargs):
                    sess = ensure_default_session()
                    self.assertTrue(get_dataflow_context().flags.is_training)
                    if training_flag:
                        self.assertTrue(sess.run(training_flag))
                    return method(*args, **kwargs)
                return inner

            for name in ('get', 'all', 'iter_epoch_batches'):
                setattr(df, name, df_wrapper(getattr(df, name)))

            # wrap the monitor to check environment
            def monitor_wrapper(method):
                @six.wraps(method)
                def inner(*args, **kwargs):
                    sess = ensure_default_session()
                    self.assertFalse(get_dataflow_context().flags.is_training)
                    if training_flag:
                        self.assertFalse(sess.run(training_flag))
                    return method(*args, **kwargs)
                return inner

            for name in ('before_training', 'start_training', 'end_training',
                         'start_epoch', 'end_epoch', 'start_step', 'end_step'):
                setattr(m, name, monitor_wrapper(getattr(m, name)))

            with tf.Session():
                run_steps(train_fn, df, batch_size=2, max_steps=3)

if __name__ == '__main__':
    unittest.main()
