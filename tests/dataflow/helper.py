# -*- coding: utf-8 -*-
import numpy as np

from madoka.utils import round_int


def dataflow_slices_generator(length, samples=100):
    # manually generate some slices
    class Gen:
        def __getitem__(self, item):
            return item
    g = Gen()
    for s in (
            # slices with non-negative positions
            g[:],
            g[0:],
            g[0: length],
            g[0::2],
            g[1::1],
            g[1::2],
            g[1: length-1: 1],
            g[1: length-1: 2],
            g[length-1:length],
            g[length-1:length:2],

            # slices with negative positions
            g[1: -1: 1],
            g[1: -1: 2],
            g[-2: -1: 1],

            # slices with negative steps
            g[-1: 1: -1],
            g[-1: 1: -2],
            g[-1: -2: -1],
            g[-1: -2: -2],
            g[-1: -3: -2],
            g[length-2:: -2],
            g[length-1:: -1],
            g[length-1:: -2],
            g[length:: -1],
            g[length:: -2],

            # these indices should result in non entries
            g[1: length: -1],
            g[length:],
            g[-2: -1: -1]):
        yield s

    # automatically generate some slices
    for j in range(samples):
        start, stop = np.random.randint(-length, length, size=2)
        step = np.random.randint(1, 10) * np.random.choice([-1, 1])
        yield slice(start, stop, step)


def dataflow_indices_generator(length, samples=100):
    # manually generate some samples
    yield np.arange(length)
    yield np.asarray([], dtype=np.int)

    # randomly generate some samples
    for j in range(samples):
        idx = np.random.randint(
            -length,
            length,
            size=np.random.randint(round_int(.5 * length),
                                   round_int(2. * length))
        )
        yield idx


def dataflow_masks_generator(length, samples=10):
    # manually generate some samples
    yield np.zeros(shape=[length]).astype(np.bool)
    yield np.ones(shape=[length]).astype(np.bool)

    # randomly generate some samples
    for j in range(samples):
        mask = np.random.binomial(1, .5, size=length)
        mask = mask.astype(np.bool)
        yield mask


def format_slice(s):
    return ':'.join(str(i) if i is not None else ''
                    for i in (s.start, s.stop, s.step))


def do_getitem_checks(arrays, df):
    def f(df_arrays, np_arrays):
        for x, y in zip(df_arrays, np_arrays):
            np.testing.assert_array_equal(x, y)

    if len(df) > 0:
        # test `.all` of data flow
        f(df.all(), [a for a in arrays])
        # test `.get`+slice of data flow
        for s2 in dataflow_slices_generator(len(df), samples=10):
            f(df.get(s2), [a[s2] for a in arrays])
        # test `.get`+indices of data flow
        for idx in dataflow_indices_generator(len(df), samples=25):
            f(df.get(idx), [a[idx] for a in arrays])
        # test `.get`+masks of data flow
        for mask in dataflow_masks_generator(len(df), samples=25):
            f(df.get(mask), [a[mask] for a in arrays])


def do_subset_all_checks(arrays, df):
    def f(df_arrays, np_arrays):
        for x, y in zip(df_arrays, np_arrays):
            np.testing.assert_array_equal(x, y)

    if len(df) > 0:
        for s2 in dataflow_slices_generator(len(df), samples=10):
            f(df[s2].all(), [a[s2] for a in arrays])
        # test `.get`+indices of data flow
        for idx in dataflow_indices_generator(len(df), samples=25):
            f(df[idx].all(), [a[idx] for a in arrays])
        # test `.get`+masks of data flow
        for mask in dataflow_masks_generator(len(df), samples=25):
            f(df[mask].all(), [a[mask] for a in arrays])
