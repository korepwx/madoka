# -*- coding: utf-8 -*-
import contextlib
import re
import unittest

import tensorflow as tf

from madoka.utils.tfcompat import GLOBAL_VARIABLES_KEY
from madoka.utils.tfhelper import (ScopedObject, root_variable_space,
                                   variable_space, root_name_space)


class _MyScopedObject(ScopedObject):

    def __init__(self, name):
        super(_MyScopedObject, self).__init__(name)


def _get_var(name, **kwargs):
    kwargs.setdefault('shape', ())
    return tf.get_variable(name, **kwargs)


def _get_op(name):
    return tf.add(1, 2, name=name)


class ScopeUnitTest(unittest.TestCase):
    """Unit tests to check the functions of scope utilities."""

    @contextlib.contextmanager
    def _assert_exception(self, exp, msg):
        with self.assertRaises(exp) as cm:
            yield
        got_msg = str(cm.exception)
        if hasattr(msg, 'match'):
            self.assertTrue(
                msg.match(got_msg),
                msg='expected message %r but got %r' % (msg, got_msg)
            )
        else:
            self.assertEquals(got_msg, msg)

    def _assert_var_exists(self, name):
        pattern = re.compile(r'^Variable (|.*/)%s already exists.*' % name)
        with self._assert_exception(ValueError, pattern):
            _get_var(name)

    def _assert_reuse_var(self, name):
        pattern = re.compile(
            r'^Trying to share variable (|.*/)%s, but specified shape.*' % name)
        with self._assert_exception(ValueError, pattern):
            _get_var(name, shape=(None,))

    def test_root_name_space(self):
        """Test the function `root_name_space`."""
        with tf.Graph().as_default():
            self.assertEquals(_get_op('o1').name, 'o1:0')
            with tf.name_scope('a') as a:
                self.assertEquals(a, 'a/')
                self.assertEquals(_get_op('o1').name, 'a/o1:0')
                with root_name_space() as root:
                    self.assertEquals(root, '')
                    self.assertEquals(_get_op('o1').name, 'o1_1:0')
                self.assertEquals(_get_op('o1').name, 'a/o1_1:0')

    def test_root_variable_space(self):
        """Test the function `root_variable_space`."""
        with tf.Graph().as_default():
            self.assertEquals(_get_var('v1').name, 'v1:0')
            self.assertEquals(_get_op('o1').name, 'o1:0')

            with tf.variable_scope('a') as a:
                self.assertEquals(a.name, 'a')
                self.assertEquals(a.original_name_scope, 'a/')
                self.assertEquals(_get_var('v1').name, 'a/v1:0')
                self.assertEquals(_get_op('o1').name, 'a/o1:0')

                with root_variable_space() as s:
                    self.assertEquals(s.name, '')
                    self.assertEquals(s.original_name_scope, '')
                    self.assertEquals(_get_var('v2').name, 'v2:0')
                    self.assertEquals(_get_op('o2').name, 'o2:0')
                    self._assert_var_exists('v1')

                    with tf.variable_scope('x') as x:
                        self.assertEquals(x.name, 'x')
                        self.assertEquals(x.original_name_scope, 'x/')
                        self.assertEquals(_get_var('v1').name, 'x/v1:0')
                        self.assertEquals(_get_op('o1').name, 'x/o1:0')

                a = tf.get_variable_scope()
                self.assertEquals(a.name, 'a')
                self.assertEquals(a.original_name_scope, 'a/')
                self.assertEquals(_get_var('v2').name, 'a/v2:0')
                self.assertEquals(_get_op('o2').name, 'a/o2:0')

                with root_variable_space(reuse=True) as s:
                    self.assertEquals(s.name, '')
                    self.assertEquals(s.original_name_scope, '')
                    self.assertEquals(_get_var('v2').name, 'v2:0')
                    self.assertEquals(_get_op('o2').name, 'o2_1:0')
                    self._assert_reuse_var('v1')

            with tf.variable_scope('b', reuse=True) as b:
                self.assertEquals(b.name, 'b')
                self.assertEquals(b.original_name_scope, 'b/')
                self.assertTrue(b.reuse)

                # test inherit settings from current name scope
                with root_variable_space():
                    self.assertEquals(_get_var('v1').name, 'v1:0')

                # test overwrite settings
                with root_variable_space(reuse=False):
                    with self.assertRaises(ValueError) as cm:
                        _get_var('v1')
                    self.assertTrue(
                        str(cm.exception).startswith(
                            'Variable v1 already exists, disallowed.'))

        # test whether or not we can open a name scope when none exists.
        with tf.Graph().as_default():
            with root_variable_space() as s:
                self.assertEquals(s.name, '')
                self.assertEquals(s.original_name_scope, '')
                self.assertEquals(_get_var('v1').name, 'v1:0')
                self.assertEquals(_get_op('o1').name, 'o1:0')

            # `root_name_space()` should empty the variable scope
            # collection after exit, so `get_variable_scope` should
            # return a new name scope.
            self.assertIsNot(s, tf.get_variable_scope())

    def test_variable_space(self):
        """Test the function `variable_space`."""
        with tf.Graph().as_default():
            # test to enter and re-enter sub scopes normally
            with variable_space('a') as a:
                self.assertEquals(a.name, 'a')
                self.assertEquals(a.original_name_scope, 'a/')
                self.assertEquals(_get_var('v1').name, 'a/v1:0')
                self.assertEquals(_get_op('o1').name, 'a/o1:0')

                with variable_space('b') as b:
                    self.assertEquals(b.name, 'a/b')
                    self.assertEquals(b.original_name_scope, 'a/b/')
                    self.assertEquals(_get_var('v1').name, 'a/b/v1:0')
                    self.assertEquals(_get_op('o1').name, 'a/b/o1:0')

                    with variable_space(a) as a2:
                        self.assertEquals(a2.name, 'a')
                        self.assertEquals(a2.original_name_scope, 'a/')
                        self.assertEquals(_get_var('v2').name, 'a/v2:0')
                        self.assertEquals(_get_op('o2').name, 'a/o2:0')

                        with variable_space('c') as c:
                            self.assertEquals(c.name, 'a/c')
                            self.assertEquals(c.original_name_scope, 'a/c/')
                            self.assertEquals(_get_var('v1').name, 'a/c/v1:0')
                            self.assertEquals(_get_op('o1').name, 'a/c/o1:0')

                a = tf.get_variable_scope()
                self.assertEquals(a.name, 'a')
                self.assertEquals(a.original_name_scope, 'a/')
                self.assertEquals(_get_var('v3').name, 'a/v3:0')
                self.assertEquals(_get_op('o3').name, 'a/o3:0')

            # test to enter sub scope with path
            with variable_space('x/y/z') as xyz:
                self.assertEquals(xyz.name, 'x/y/z')
                self.assertEquals(xyz.original_name_scope, 'x/y/z/')
                xyz_v1 = _get_var('v1')
                self.assertEquals(xyz_v1.name, 'x/y/z/v1:0')
                self.assertEquals(_get_op('o1').name, 'x/y/z/o1:0')

            with variable_space('x/y/w') as xyw:
                self.assertEquals(xyw.name, 'x/y/w')
                self.assertEquals(xyw.original_name_scope, 'x/y/w/')
                xyw_v1 = _get_var('v1')
                self.assertEquals(xyw_v1.name, 'x/y/w/v1:0')
                self.assertEquals(_get_op('o1').name, 'x/y/w/o1:0')

            self.assertEquals(
                tf.get_collection(GLOBAL_VARIABLES_KEY, scope='x'),
                [xyz_v1, xyw_v1]
            )
            self.assertEquals(
                tf.get_collection(GLOBAL_VARIABLES_KEY, scope='x/y'),
                [xyz_v1, xyw_v1]
            )
            self.assertEquals(
                tf.get_collection(GLOBAL_VARIABLES_KEY, scope='x/y/z'),
                [xyz_v1]
            )
            self.assertEquals(
                tf.get_collection(GLOBAL_VARIABLES_KEY, scope='x/y/w'),
                [xyw_v1]
            )

            # test to re-enter the root scope
            root = tf.get_variable_scope()
            self.assertEquals(root.name, '')
            self.assertEquals(root.original_name_scope, '')
            self.assertEquals(_get_var('v1').name, 'v1:0')
            self.assertEquals(_get_op('o1').name, 'o1:0')

            with variable_space(xyz) as s:
                self.assertEquals(s.name, 'x/y/z')
                self.assertEquals(s.original_name_scope, 'x/y/z/')

                with variable_space(root) as ss:
                    self.assertEquals(ss.name, '')
                    self.assertEquals(ss.original_name_scope, '')
                    self.assertEquals(_get_var('v2').name, 'v2:0')
                    self.assertEquals(_get_op('o2').name, 'o2:0')

                    with variable_space(xyw) as sss:
                        self.assertEquals(sss.name, 'x/y/w')
                        self.assertEquals(sss.original_name_scope, 'x/y/w/')
                        self.assertEquals(_get_var('v2').name, 'x/y/w/v2:0')
                        self.assertEquals(_get_op('o2').name, 'x/y/w/o2:0')

                    ss = tf.get_variable_scope()
                    self.assertEquals(ss.name, '')
                    self.assertEquals(ss.original_name_scope, '')

                s = tf.get_variable_scope()
                self.assertEquals(s.name, 'x/y/z')
                self.assertEquals(s.original_name_scope, 'x/y/z/')

            # test to re-enter a deep scope.
            with variable_space(c) as s:
                self.assertEquals(s.name, 'a/c')
                self.assertEquals(s.original_name_scope, 'a/c/')

                with variable_space(xyz) as ss:
                    self.assertEquals(ss.name, 'x/y/z')
                    self.assertEquals(ss.original_name_scope, 'x/y/z/')
                    self.assertEquals(_get_var('v2').name, 'x/y/z/v2:0')
                    self.assertEquals(_get_op('o2').name, 'x/y/z/o2:0')

                self.assertEquals(s.name, 'a/c')
                self.assertEquals(s.original_name_scope, 'a/c/')

            # test to overwrite the scope settings.
            with variable_space(c) as s:
                self.assertEquals(s.name, 'a/c')
                self.assertEquals(s.original_name_scope, 'a/c/')
                self._assert_var_exists('v1')
                self.assertEquals(_get_op('o1').name, 'a/c/o1_1:0')

                with variable_space(s, reuse=True):
                    self.assertEquals(_get_var('v1').name, 'a/c/v1:0')
                    self.assertEquals(_get_op('o1').name, 'a/c/o1_2:0')
                    self._assert_reuse_var('v1')

    def test_ScopedObject(self):
        with tf.Graph().as_default():
            # test to create a scoped object within variable scope
            with tf.variable_scope('a') as p:
                self.assertEquals(p.name, 'a')
                obj = _MyScopedObject('b')
                self.assertEquals(obj.scope.name, 'a/b')

            # try to create variables and operations withinthe variable scope
            with obj.variable_space():
                self.assertEquals(_get_var('v1').name, 'a/b/v1:0')
                self.assertEquals(_get_op('o1').name, 'a/b/o1:0')

            # try to create variables and operations within sub variable scope
            with obj.variable_space('c/d'):
                self.assertEquals(_get_var('v1').name, 'a/b/c/d/v1:0')
                self.assertEquals(_get_op('o1').name, 'a/b/c/d/o1:0')
