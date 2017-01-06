# -*- coding: utf-8 -*-
import six

from madoka.utils import TypedConfigAttr, DefaultConfigParser, StrConfigAttr

__all__ = ['config']


class FloatXConfigAttr(TypedConfigAttr):
    def __init__(self, name, description):
        def filter(val):
            if isinstance(val, six.string_types):
                val = val.lower()
            if val in ('float32', '32'):
                return 'float32'
            elif val in ('float64', '64'):
                return 'float64'

        def valid(val):
            return val is not None

        super(FloatXConfigAttr, self).__init__(name, 'float32', filter, valid,
                                               description)


class MadokaConfig(object):
    """Madoka config object."""

    floatX = FloatXConfigAttr(
        'floatX',
        description='default type of the float numbers'
    )
    log_format = StrConfigAttr(
        'log_format',
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        description='default log format'
    )
    log_level = StrConfigAttr(
        'log_level',
        'INFO',
        description='default log level'
    )
    train_lock_zk = StrConfigAttr(
        'train_lock_zk',
        None,
        description='URI of the ZooKeeper server, from where the training '
                    'file locks should be acquired.'
    )

config = MadokaConfig()
"""Global configuration of Madoka package."""


def _init():
    DefaultConfigParser('madoka').load(config)

    import sys
    del sys.modules[__name__]._init

_init()
