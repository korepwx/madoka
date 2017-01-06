# -*- coding: utf-8 -*-

import re

import tensorflow as tf


# detect the version of TensorFlow
tf_ver = tuple([
    int(v)
    for v in re.match(r'^(\d+(?:\.\d+)*)', tf.__version__).group(1).split('.')
])
