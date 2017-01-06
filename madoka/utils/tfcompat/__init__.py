# -*- coding: utf-8 -*-
"""Utilities for compatibility across different versions of TensorFlow."""

# import the symbols according to TensorFlow version
from ._ver import tf_ver
if tf_ver >= (0, 12):
    from ._0_12 import *
elif tf_ver >= (0, 11):
    from ._0_11 import *
else:
    raise RuntimeError('TensorFlow < 0.11 is not supported.')


# export the symbols
__all__ = [
    'GLOBAL_VARIABLES_KEY', 'variables_initializer', 'global_variables',
    'global_variables_initializer', 'histogram_summary', 'scalar_summary',
    'merge_summary', 'SummaryWriter',
]
