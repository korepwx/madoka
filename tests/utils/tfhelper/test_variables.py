# -*- coding: utf-8 -*-
import unittest

import tensorflow as tf

from madoka.utils import tfhelper
from madoka.utils.tfcompat import GLOBAL_VARIABLES_KEY


class VariablesTestCase(unittest.TestCase):
    """Unit tests for variables utility."""

    def test_selector(self):
        """Test `VariableSelector()`."""
        def get_var(name, trainable=False, collections=None):
            collections = collections or []
            if trainable:
                collections.append(tf.GraphKeys.TRAINABLE_VARIABLES)
            collections.append(GLOBAL_VARIABLES_KEY)
            return tf.get_variable(name, initializer=0, dtype=tf.int32,
                                   trainable=trainable, collections=collections)

        def check(sel, varlist):
            mismatch_msg = '\n  {%s} != {%s}' % (
                ', '.join(sorted(v.name for v in sel.select())),
                ', '.join(sorted(v.name for v in varlist))
            )
            mismatch_msg += '\ncaused by:\n  %s' % sel
            self.assertEquals(set(sel.select()), varlist, msg=mismatch_msg)

        with tf.Graph().as_default():
            v1 = get_var('v1')
            v2 = get_var('v2', trainable=True)
            v3 = get_var('v3', trainable=True,
                         collections=[tfhelper.GraphKeys.TRAINING_STATES])
            v4 = get_var('v4', collections=[tfhelper.GraphKeys.TRAINING_STATES,
                                            'something'])
            v5 = get_var('v5', collections=[tfhelper.GraphKeys.TRAINER_SLOTS,
                                            'something'])

            from madoka.utils.tfhelper import VariableSelector as vs

            # test individual selectors.
            check(vs.all(), {v1, v2, v3, v4, v5})
            check(vs.list([v1, v3, v4]), {v1, v3, v4})
            check(vs.collection('something'), {v4, v5})
            check(vs.trainable(), {v2, v3})
            check(vs.training_states(), {v3, v4})
            check(vs.trainer_slots(), {v5})

            # test composition of selectors.
            check(
                vs.training_states() & vs.trainable() & vs.list([v1, v3, v4]),
                {v3}
            )
            check(
                vs.training_states() & (vs.trainable() & vs.list([v1, v3, v4])),
                {v3}
            )
            check(
                vs.trainable() | vs.training_states() | vs.trainer_slots(),
                {v2, v3, v4, v5}
            )
            check(
                vs.trainable() | (vs.training_states() | vs.trainer_slots()),
                {v2, v3, v4, v5}
            )
            check(
                vs.trainable() + vs.training_states() + vs.trainer_slots(),
                {v2, v3, v4, v5}
            )
            check(
                vs.trainable() + (vs.training_states() + vs.trainer_slots()),
                {v2, v3, v4, v5}
            )
            check(-vs.trainable(), {v1, v4, v5})
            check(--vs.trainable(), {v2, v3})
            check(
                (vs.trainable() | vs.training_states()) & -vs.list([v1, v4]),
                {v2, v3}
            )
            check(
                vs.all() - vs.trainable() - vs.training_states(),
                {v1, v5}
            )
            check(
                vs.all() - (vs.trainable() + vs.training_states()),
                {v1, v5}
            )
            check(
                vs.all() - (vs.trainable() - vs.training_states()),
                {v1, v3, v4, v5}
            )
            check(-vs.trainable() - vs.trainer_slots(), {v1, v4})
            check((-vs.trainable()) & vs.trainer_slots(), {v5})
            check(
                (vs.list([v1, v2, v3]) - vs.training_states()) -
                (vs.trainable() - vs.training_states()),
                {v1}
            )
