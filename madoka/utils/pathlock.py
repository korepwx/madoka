# -*- coding: utf-8 -*-

"""Utility to lock a path on local file system.

In single-node situations, it is enough to lock a path on local file system
by using FileLock.  However, when running in a multi-node cluster with shared
file system, the plain FileLock might not work well.

This package provides two ways to lock a path on the file system: either by
using the plain FileLock, or by using ZooKeeper.
"""
import six
from filelock import FileLock
from kazoo.client import KazooClient
from kazoo.recipe.lock import Lock as KazooLock

if six.PY2:
    from urlparse import urlparse
else:
    from urllib.parse import urlparse

__all__ = ['PathLock']


class PathLock(object):
    """Common interface for path locks.

    Parameters
    ----------
    path : str
        The path which should be locked.

        With FileLock, this will be used as the path of lock file.
        With ZooKeeper, this will be the path of the node.

    timeout : int
        Timeout for acquiring the lock.
        If not specified, will attempt to lock for infinite amount of time.
    """

    def __init__(self, path, timeout=None):
        self.path = path
        self.timeout = timeout

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.path)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def acquire(self):
        """Acquire the lock."""
        raise NotImplementedError()

    def release(self):
        """Release the lock."""
        raise NotImplementedError()

    @staticmethod
    def plain(path, timeout=None):
        """Open a plain FileLock.

        Parameters
        ----------
        path : str
            Path to be locked.

        timeout : int
            Timeout for acquiring the lock.

        Returns
        -------
        PathLock
        """
        return _FileLock(path, timeout=timeout)

    @staticmethod
    def zookeeper(uri, path, timeout=None):
        """Open a Zookeeper lock.

        Parameters
        ----------
        uri : str
            URI of the ZooKeeper server, e.g., "zk://127.0.0.1:2121/root"

        path : str
            Path to be locked.

        timeout : int
            Timeout for acquiring the lock.

        Returns
        -------
        PathLock
        """
        return _ZooKeeperLock(uri, path, timeout=timeout)


class _FileLock(PathLock):
    """Path lock backed by plain FileLock."""

    def __init__(self, path, timeout=None):
        super(_FileLock, self).__init__(path, timeout=timeout)
        self._lock = FileLock(path)

    def acquire(self):
        self._lock.acquire()

    def release(self):
        self._lock.release()


class _ZooKeeperLock(PathLock):
    """Path lock backed by ZooKeeper."""

    def __init__(self, uri, path, timeout=None):
        # check arguments
        u = urlparse(uri)
        if u.scheme and u.scheme != 'zk':
            raise ValueError('%r is not a ZooKeeper uri.' % uri)
        if not u.netloc:
            raise ValueError('Empty server in %r.' % uri)
        zk_hosts = u.netloc
        zk_root = u.path.rstrip('/')
        if not zk_root.startswith('/'):
            zk_root = '/' + zk_root
        path = path.rstrip('/')
        if not path.startswith('/'):
            path = '/' + path

        # memorize arguments
        super(_ZooKeeperLock, self).__init__(path, timeout=timeout)
        self._hosts = zk_hosts
        self._root = zk_root

        # lazy initialized members
        self._zk = None  # type: KazooClient
        self._lock = None  # type: KazooLock

    def acquire(self):
        self._zk = KazooClient(hosts=self._hosts)
        self._lock = KazooLock(self._zk, self._root + self.path)
        self._zk.start()
        self._lock.acquire(timeout=self.timeout)

    def release(self):
        self._lock.release()
        self._zk.stop()
        self._zk.close()
        self._lock = None
        self._zk = None
