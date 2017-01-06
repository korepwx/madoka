# -*- coding: utf-8 -*-
"""Utilities for composing neural networks with TensorFlow.

The purpose of this package is not to establish a comprehensive, standalone
framework for composing neural network models.  Instead it is designed to be
compatible with other libraries like TFLearn and PrettyTensor, and even with
TensorFlow alone.
"""

from . import init
from . import metric
from . import nonlinearity
from .layer import *
from .op import *
