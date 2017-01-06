# -*- coding: utf-8 -*-

import tensorflow as tf

# symbol aliases
GLOBAL_VARIABLES_KEY = tf.GraphKeys.VARIABLES

variables_initializer = tf.initialize_variables
global_variables = tf.all_variables
global_variables_initializer = tf.initialize_all_variables
histogram_summary = tf.histogram_summary
scalar_summary = tf.scalar_summary
merge_summary = tf.merge_summary
SummaryWriter = tf.train.SummaryWriter
