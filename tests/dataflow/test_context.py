# -*- coding: utf-8 -*-
import unittest

from madoka.dataflow import (DataFlowContext, get_dataflow_context,
                             DataFlow)
from madoka.utils import TrainingPhase


class DataFlowContextTestCase(unittest.TestCase):
    """Unit tests for data flow contexts."""

    def test_as_default(self):
        ctx = DataFlowContext([])
        ctx2 = DataFlowContext([])
        self.assertIsNone(get_dataflow_context())
        with ctx.as_default():
            self.assertIs(get_dataflow_context(), ctx)
            with ctx2.as_default():
                self.assertIs(get_dataflow_context(), ctx2)
            self.assertIs(get_dataflow_context(), ctx)
        self.assertIsNone(get_dataflow_context())

    def test_set_flags(self):
        ctx = DataFlowContext([])
        self.assertFalse(ctx.flags.is_training)
        with ctx.as_default(), ctx.set_flags(phase=TrainingPhase.TRAINING):
            self.assertTrue(ctx.flags.is_training)
            with ctx.set_flags(phase=TrainingPhase.NOT_SET):
                self.assertFalse(ctx.flags.is_training)
            self.assertTrue(ctx.flags.is_training)
        self.assertFalse(ctx.flags.is_training)

    def test_topological_order(self):
        class GraphNode(DataFlow):
            def __init__(self, name, parents=None):
                self.name = name
                if isinstance(parents, DataFlow):
                    parents = [parents]
                self._parents = tuple(parents or [])

            @property
            def input_flows(self):
                return self._parents

            def __repr__(self):
                return self.name

        root1 = GraphNode('a')
        root2 = GraphNode('b')
        level1_1 = GraphNode('c', [root1, root2])
        level1_2 = GraphNode('d', root2)
        level2_1 = GraphNode('e', level1_1)
        level2_2 = GraphNode('f', level1_1)
        level2_3 = GraphNode('g', level1_2)
        level3 = GraphNode('h', [level2_3, level2_2, level2_1])

        ctx = DataFlowContext(level3)
        self.assertEquals(
            ctx.all_flows,
            (root2, level1_2, level2_3, root1, level1_1, level2_2, level2_1,
             level3)
        )
