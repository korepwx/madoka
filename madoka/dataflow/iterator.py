# -*- coding: utf-8 -*-

"""Utitilies for iterating through a DataFlow."""

import copy
import multiprocessing
import os
import signal
import sys

import numpy as np
import sharedmem as sm
import six

from .base import DataFlow
from .context import DataFlowContext, DataFlowFlags, get_dataflow_context

if six.PY2:
    import Queue as queue
else:
    import queue

__all__ = ['dataflow_batch_iterator']


def dataflow_batch_iterator(dataflow, batch_size, prefetch_num=None,
                            ignore_incomplete_batch=False, context=None,
                            flags=None, loop_epoch=False):
    """Create a DataFlow batch iterator.

    Parameters
    ----------
    dataflow : DataFlow
        The data flow to be iterated through.

    batch_size : int
        Size of each batch from this iterator.

    prefetch_num : int
        If specified, will prefetch this number of batches in background worker.
        Must be at least 2.

    ignore_incomplete_batch : bool
        Whether or not to ignore the final batch if it contains less
        than ``batch-size`` number of items?

    context : DataFlowContext
        The data flow context.
        If not specified, will use the current data flow context.

    flags : DataFlowFlags
        Additional flags passed to data flow context.

    loop_epoch : bool
        Whether or not to loop through different epochs? (Default False)

    Returns
    -------
    DataFlowBatchIterator
        The batch iterator of given data flow.
    """
    if prefetch_num is not None:
        ret = _AsyncBatchIterator(
            dataflow=dataflow,
            batch_size=batch_size,
            ignore_incomplete_batch=ignore_incomplete_batch,
            prefetch_num=prefetch_num,
            context=context,
            flags=flags,
            loop_epoch=loop_epoch
        )
    else:
        ret = _SyncBatchIterator(
            dataflow=dataflow,
            batch_size=batch_size,
            ignore_incomplete_batch=ignore_incomplete_batch,
            context=context,
            flags=flags,
            loop_epoch=loop_epoch
        )
    return ret


class DataFlowBatchIterator(object):
    """Base class for all DataFlow batch iterators.

    Parameters
    ----------
    dataflow : DataFlow
        The data flow to be iterated through.

    batch_size : int
        Size of each batch from this iterator.

    ignore_incomplete_batch : bool
        Whether or not to ignore the final batch if it contains less
        than ``batch-size`` number of items?

    context : DataFlowContext
        The data flow context.
        If not specified, will use the current data flow context.

    flags : DataFlowFlags
        Additional flags passed to data flow context.

    loop_epoch : bool
        Whether or not to loop through different epochs? (Default False)
    """

    def __init__(self, dataflow, batch_size, ignore_incomplete_batch=False,
                 context=None, flags=None, loop_epoch=False):
        self.dataflow = dataflow
        self.batch_size = batch_size
        self.ignore_incomplete_batch = ignore_incomplete_batch
        self.context = context
        self.flags = flags
        self.loop_epoch = loop_epoch

    def __iter__(self):
        """Iterate through the data flow.

        Yields
        ------
        (int, tuple[numpy.ndarray])
            Tuple of (epoch_index, batch_arrays).

            If ``loop_epoch`` is False, the epoch index will always be zero.

        Notes
        -----
        The ``reset_epoch`` will always be called AFTER each epoch, but will
        NOT be called BEFORE the first epoch.
        """
        raise NotImplementedError()


class _SyncBatchIterator(DataFlowBatchIterator):
    """Synchronous DataFlow batch iterator."""

    def __iter__(self):
        # detect the context and the flags
        context = self.context
        if context is None:
            context = get_dataflow_context()
            if context is None:
                raise ValueError('No data flow context has been open.')
        flags = copy.copy(context.flags)
        if self.flags:
            flags.merge_from(self.flags)

        # actually iterate through the data
        with context.set_flags(**flags.as_dict()):
            if self.loop_epoch:
                epoch = 0
                while True:
                    for batch in self.dataflow.iter_epoch_batches(
                            self.batch_size, self.ignore_incomplete_batch):
                        yield epoch, batch
                    context.reset_epoch()
                    epoch += 1
            else:
                for batch in self.dataflow.iter_epoch_batches(
                        self.batch_size, self.ignore_incomplete_batch):
                    yield 0, batch
                context.reset_epoch()


class _AsyncBatchIteratorChildProcessError(Exception):
    """Exception from child process of an async batch iterator."""

    def __init__(self, cause):
        self.cause = cause

    def __cause__(self):
        return self.cause


class _AsyncBatchIterator(DataFlowBatchIterator):
    """Asynchronous DataFlow batch iterator.

    This iterator will prefetch batches in a background worker, and push
    data to the foreground program through a pipe or queue.
    The implementation of this iterator may vary on different platforms.

    Parameters
    ----------
    dataflow : DataFlow
        The data flow to be iterated through.

    batch_size : int
        Size of each batch from this iterator.

    prefetch_num : int
        Number of batches to prefetch (must be at least 2).

    ignore_incomplete_batch : bool
        Whether or not to ignore the final batch if it contains less
        than ``batch-size`` number of items?

    context : DataFlowContext
        The data flow context.
        If not specified, will use the current data flow context.

    flags : DataFlowFlags
        Additional flags passed to data flow context.

    loop_epoch : bool
        Whether or not to loop through different epochs? (Default False)
    """

    def __init__(self, dataflow, batch_size, prefetch_num,
                 ignore_incomplete_batch=False, context=None,
                 flags=None, loop_epoch=False):
        if not isinstance(prefetch_num, six.integer_types) or \
                prefetch_num <= 1:
            raise TypeError('`prefetch_num` must be an integer >= 2.')
        super(_AsyncBatchIterator, self).__init__(
            dataflow=dataflow,
            batch_size=batch_size,
            ignore_incomplete_batch=ignore_incomplete_batch,
            context=context,
            flags=flags,
            loop_epoch=loop_epoch
        )
        self.prefetch_num = prefetch_num

    def _create_shared_arrays(self):
        bufsize = self.batch_size * self.prefetch_num
        data_types = self.dataflow.data_types
        data_shapes = tuple((bufsize,) + s for s in self.dataflow.data_shapes)
        return tuple(sm.empty(s, t) for s, t in zip(data_shapes, data_types))

    def _child_proc(self, shared_arrays, left, right):
        # Initialize a cyclic buffer using the shared array.
        batch_size = self.batch_size
        buf_size = self.prefetch_num
        buf = shared_arrays
        buf_write = 0   # next batch index for child to write

        # The flag whether or not we're interrupted
        interrupted = False

        # Prefetch data and put items into the queue
        for epoch, arrays in _SyncBatchIterator(
                dataflow=self.dataflow,
                batch_size=self.batch_size,
                ignore_incomplete_batch=self.ignore_incomplete_batch,
                context=self.context,
                flags=self.flags,
                loop_epoch=self.loop_epoch):
            # store the data of this batch into shared memory
            start = buf_write * batch_size
            array_size = len(arrays[0])
            for i, a in enumerate(arrays):
                assert (len(a) == array_size)
                assert (len(a) <= batch_size)
                buf[i][start: start + array_size] = a

            # put the item into the queue
            left.put(('D', epoch, array_size))

            # try to get command from the parent
            try:
                cmd = right.get_nowait()
                if cmd[0] == 'K':
                    interrupted = True
            except queue.Empty:
                pass

            # exit the loop if we have been interrupted
            if interrupted:
                break

            # move to next batch
            buf_write = (buf_write + 1) % buf_size

        if not interrupted:
            # Put end mark into the queue
            left.put(('X',))

            # Wait for parent to let us exit.
            assert(right.get()[0] == 'K')

    def _parent_proc(self, shared_arrays, left, right, child_pid):
        # Initialize a cyclic buffer using the shared array.
        batch_size = self.batch_size
        buf_size = self.prefetch_num
        buf = shared_arrays
        buf_read = 0  # next batch index for parent to read

        # The local array buffer for receiving data
        data_types = self.dataflow.data_types
        local_array = tuple(
            np.empty((batch_size,) + s, t)
            for s, t in zip(self.dataflow.data_shapes, data_types)
        )

        # Now receive data from the inbox
        ret = [None] * len(local_array)
        while True:
            received = left.get()
            if received[0] == 'D':
                # data received, copy to local array and yield it
                epoch, array_size = received[1:]
                start = buf_read * batch_size
                for i, a in enumerate(local_array):
                    a[: array_size] = buf[i][start: start + array_size]
                    ret[i] = a[: array_size]
                    ret[i].setflags(write=False)
                yield epoch, tuple(ret)

                # move to next batch
                buf_read = (buf_read + 1) % buf_size

            elif received[0] == 'E':
                # child process error received
                raise _AsyncBatchIteratorChildProcessError(received[1])

            elif received[0] == 'X':
                # child process reported no more data, tell the child
                # to exit and break the loop.
                right.put(('K',))
                break

            else:
                raise ValueError(
                    'Recived unrecognized data from child process: %r.' %
                    received
                )

    def __iter__(self):
        # Get the shared arrays across parent & child process.
        shared_arrays = self._create_shared_arrays()

        # Now create the queue between parent & child process.
        #
        # The parent should read from left and write to right, while
        # the child should read from right and write to left.
        left = multiprocessing.Queue(self.prefetch_num - 1)
        right = multiprocessing.Queue()

        # Fork the child process.
        pid = os.fork()
        if pid == 0:
            # Do the child process jobs.
            try:
                self._child_proc(shared_arrays, left, right)
            except Exception as ex:
                # we should send the exception to parent.
                left.put(('E', ex))
            finally:
                # Consume all remaining items in child inbox.
                while not right.empty():
                    right.get()

                # Attempt to kill the child process with syscall,
                # so that to avoid exiting any Python context.
                os.kill(os.getpid(), signal.SIGKILL)

                # If os.kill does not take effect, we should make sure
                # the process is interrupted.
                sys.exit(0)
        else:
            # Do the parent process jobs.
            try:
                for epoch, arrays in self._parent_proc(
                        shared_arrays, left, right, pid):
                    yield epoch, arrays
            finally:
                # Always notify the child to exit.
                # This won't have side effect if it's not necessary.
                try:
                    right.put(('K',))
                except Exception:
                    pass

                # Consume all remaining items in parent inbox.
                while not left.empty():
                    left.get()

                # Wait for the child process to exit.
                os.waitpid(pid, 0)
