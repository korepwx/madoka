# -*- coding: utf-8 -*-
import os

import click
import six

from madoka import config
from .server import MadokaBoardServer

__all__ = ['madoka_board']


def _init_log(log_file, log_level):
    import logging.config
    config_dict = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': config.log_format
            },
        },
        'handlers': {
            'default': {
                'level': log_level,
                'formatter': 'default',
                'class': 'logging.StreamHandler',
            },
        },
        'loggers': {
            '': {
                'handlers': ['default'],
                'level': log_level,
                'propagate': True
            },
            'tensorflow': {
                'handlers': ['default'],
                'level': 'WARN',
                'propagate': True
            },
        }
    }
    if log_file:
        config_dict['handlers']['logfile'] = {
            'level': log_level,
            'formatter': 'default',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.realpath(log_file),
            'maxBytes': 1048576
        }
        for logger in six.itervalues(config_dict['loggers']):
            logger['handlers'].append('logfile')
    logging.config.dictConfig(config_dict)


@click.command()
@click.option('-I', '--interval', default=120,
              help='Number of seconds between two successive scans of '
                   'training storage.')
@click.option('-i', '--ip', default='', help='HTTP server IP.')
@click.option('-p', '--port', default=8080, help='HTTP server port.')
@click.option('-l', '--log-file', default=None, help='Save log to file.')
@click.option('-L', '--log-level', default=config.log_level, help='Log level.')
@click.option('-s', '--storage', multiple=True,
              help='Map a URL prefix to training storage. '
                   'For example, "-s /foo:/path/to/foo".')
@click.option('--chown', default=False, is_flag=True,
              help='Chown storage and tags to that of the parent.')
@click.option('--debug', default=False, is_flag=True,
              help='Whether or not to enable debugging features?')
def madoka_board(interval, ip, port, log_file, log_level, storage, chown,
                 debug):
    """HTTP server for Madoka training storage browser."""
    # parse the storage mappings
    storage_root = {}
    for s in storage:
        arr = s.split(':', 1)
        if len(arr) != 2:
            raise ValueError('%r: unrecognized mapping.' % (s,))
        prefix, path = tuple(arr)
        prefix = '/' + prefix.strip('/')
        path = os.path.realpath(path)
        storage_root[prefix] = path

    # initialize the logging
    _init_log(log_file, log_level)

    # start the server
    server = MadokaBoardServer(
        storage_root=storage_root,
        scan_interval=interval,
        interface=ip,
        port=port,
        chown_to_parent=chown,
        debug=debug
    )
    server.run()
