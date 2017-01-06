# -*- coding: utf-8 -*-
import unittest

import numpy as np
from sklearn.preprocessing import StandardScaler

from madoka.dataflow import DataFlow


class TransformTestCase(unittest.TestCase):
    """Unit tests for transforming data flows."""

    def test_transform(self):
        """Test transformed data flows."""
        arrays = [np.arange(1000, dtype=np.int),
                  np.arange(1000, 2000, dtype=np.int)]
        df = DataFlow.from_numpy(arrays)

        # test applying transform on the first array
        df2 = df.apply(lambda v: v+123)
        self.assertEquals(df2.array_count, 2)
        self.assertEquals(len(df2), 1000)
        self.assertEquals(df2.data_shapes, ((), ()))
        self.assertEquals(df2.data_types, (np.int, np.int))
        np.testing.assert_array_equal(df2.all()[0], arrays[0] + 123)
        np.testing.assert_array_equal(df2.all()[1], arrays[1])

        # test applying transform on all arrays
        df3 = df.apply([
            (lambda v: np.asarray(v+123., dtype=np.float64)),
            (lambda v: v+456)
        ])
        self.assertEquals(df3.data_shapes, ((), ()))
        self.assertEquals(df3.data_types, (np.float64, np.int))
        np.testing.assert_array_equal(df3.all()[0], arrays[0] + 123)
        np.testing.assert_array_equal(df3.all()[1], arrays[1] + 456)

        # test scikit-learn transformer
        arrays = [arrays[0].astype(np.float64), arrays[1]]
        df = DataFlow.from_numpy(arrays)
        scaler = StandardScaler()
        scaler.fit(arrays[0].reshape([-1, 1]))
        arrays_transformed = \
            scaler.transform(arrays[0].reshape([-1, 1])).reshape([-1])
        df4 = df.apply(scaler)
        self.assertEquals(df4.data_shapes, ((), ()))
        self.assertEquals(df4.data_types, (np.float64, np.int))
        np.testing.assert_array_equal(df4.all()[0], arrays_transformed)

        # test other shapes
        array = np.arange(1000).reshape([-1, 2])
        df = (DataFlow.from_numpy(array).
              apply(lambda v: np.concatenate([v, v], axis=1)))
        self.assertEquals(df.data_shapes, ((4,),))
        self.assertEquals(df.data_types, (np.int,))
        np.testing.assert_array_equal(
            df.all()[0], np.concatenate([array, array], axis=1))


if __name__ == '__main__':
    unittest.main()
