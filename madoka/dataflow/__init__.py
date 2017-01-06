# -*- coding: utf-8 -*-

"""Package for ingesting and preprocessing data.

Most of the utilities provided by this package are for general purpose,
rather than for TensorFlow only.  Thus the TensorFlow pipeline may not
be supported well.  If you care about the performance of data processing,
you may need to use TensorFlow utilities to compose your pipeline.
"""

from .arraylike import *
from .base import *
from .cache import *
from .context import *
from .empty import *
from .in_memory import *
from .iterator import *
from .merge import *
from .random import *
from .subset import *
from .transform import *
from .window import *
