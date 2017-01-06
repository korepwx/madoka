# -*- coding: utf-8 -*-

import time
import unittest

import numpy as np
import six

from madoka.dataflow import (DataFlow, PipelineDataFlow, DataFlowContext,
                             dataflow_batch_iterator)


class _DelayedDataFlow(PipelineDataFlow):
    """Data flow which delays `get()` method of underlying data flow."""

    def __init__(self, origin, delay=0.1):
        super(_DelayedDataFlow, self).__init__(origin)
        self.delay = delay

    def get(self, item, array_indices=None):
        time.sleep(self.delay)
        return self._origin.get(item, array_indices)

    def all(self):
        return self._origin.all()


class IteratorTestCase(unittest.TestCase):
    """Test case for data flow iterators."""

    def test_batch_iterator(self):
        """Test case for batch iterators."""
        delay = 0.1
        df = DataFlow.from_numpy([np.arange(12).reshape([6, 2]),
                                  np.arange(6)])
        df = _DelayedDataFlow(df, delay=delay)

        def run_iterator(iterator, max_epoch=2):
            with DataFlowContext(df).as_default():
                ret = []
                for epoch, batch in iterator:
                    if epoch >= max_epoch:
                        break
                    ret.append((epoch, batch))
                    time.sleep(delay)
                return ret

        def format_result(x):
            print(x)
            return '\n'.join([
                'epoch %s:\n  %s' % (
                    epoch, '\n  '.join(repr(a) for a in arrays)
                )
                for epoch, arrays in x
            ])

        def do_test(expected, batch_size, prefetch_num, ignore_incomplete_batch,
                    loop_epoch):
            args = (
                'batch_size: %r, prefetch_num: %r, ignore_incomplete_batch: %r,'
                ' loop_epoch: %r' %
                (batch_size, prefetch_num, ignore_incomplete_batch, loop_epoch)
            )
            iterator = dataflow_batch_iterator(
                df,
                batch_size=batch_size,
                prefetch_num=prefetch_num,
                ignore_incomplete_batch=ignore_incomplete_batch,
                loop_epoch=loop_epoch
            )
            if ignore_incomplete_batch:
                batch_num = len(df) // batch_size
            else:
                batch_num = (len(df) + batch_size - 1) // batch_size
            expected_async_time = delay * batch_num * (int(loop_epoch) + 1)

            start_time = time.time()
            result = run_iterator(iterator)
            end_time = time.time()

            np.testing.assert_equal(
                result,
                expected,
                err_msg='\n'.join([
                    '  Args: %s' % args,
                    '  Results:',
                    '    ' + '\n    '.join(format_result(result).split('\n')),
                    '  Expected:',
                    '    ' + '\n    '.join(format_result(expected).split('\n'))
                ])
            )

            if prefetch_num is not None:
                self.assertLessEqual(
                    end_time - start_time,
                    1.2 * expected_async_time,
                    msg='Not seem to be asynchronous, args: %s' % args
                )
            else:
                self.assertGreaterEqual(
                    end_time - start_time,
                    1.8 * expected_async_time,
                    msg='Not seem to be synchronous, args: %s' % args
                )

        # test answers
        results = {
            (2, False, False): [
                (0, ([[0, 1], [2, 3]], [0, 1])),
                (0, ([[4, 5], [6, 7]], [2, 3])),
                (0, ([[8, 9], [10, 11]], [4, 5])),
            ],
            (5, False, False): [
                (0, ([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
                     [0, 1, 2, 3, 4])),
                (0, ([[10, 11]], [5])),
            ],
            (6, False, False): [
                (0, ([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
                     [0, 1, 2, 3, 4, 5])),
            ],
        }
        for k, v in list(six.iteritems(results)):
            if len(v[-1][1][0]) != k[0]:
                v = v[:-1]
            results[(k[0], True, False)] = v
        for k, v in list(six.iteritems(results)):
            v2 = v + [(1,) + x[1:] for x in v]
            results[(k[0], k[1], True)] = v2

        # test cases
        for batch_size in (2, 5, 6):
            for prefetch_num in (None,):
                for ignore_incomplete_batch in (False, True):
                    for loop_epoch in (False, True):
                        res = results[(batch_size, ignore_incomplete_batch,
                                       loop_epoch)]
                        do_test(
                            res,
                            batch_size=batch_size,
                            prefetch_num=prefetch_num,
                            ignore_incomplete_batch=ignore_incomplete_batch,
                            loop_epoch=loop_epoch
                        )
