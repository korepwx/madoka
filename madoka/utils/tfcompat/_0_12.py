# -*- coding: utf-8 -*-

import tensorflow as tf

# symbol aliases
GLOBAL_VARIABLES_KEY = tf.GraphKeys.GLOBAL_VARIABLES

variables_initializer = tf.variables_initializer
global_variables = tf.global_variables
global_variables_initializer = tf.global_variables_initializer
histogram_summary = tf.summary.histogram
scalar_summary = tf.summary.scalar
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter
