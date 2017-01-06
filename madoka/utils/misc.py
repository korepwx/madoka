# -*- coding: utf-8 -*-
import contextlib
import io
import json
import os
import random
import subprocess
import sys
import time
import uuid

import numpy as np
import six

__all__ = [
    'AutoJsonEncoder', 'assert_raises_message', 'atomic_replace',
    'check_argtype', 'ensure_dir_exists', 'ensure_parent_exists',
    'fileno_redirected', 'filter_not_none', 'flatten_list',
    'get_process_identity', 'init_logging', 'is_mask_array', 'round_int',
    'safe_overwrite_file', 'set_current_directory', 'unique', 'where',
    'wrap_text_writer', 'write_string', 'yield_parent_dirs',
    'duplicate_console_output',
]


def unique(iterable):
    """Uniquify elements to construct a new list.

    Parameters
    ----------
    iterable : collections.Iterable[any]
        The collection to be uniquified.

    Returns
    -------
    list[any]
        Unique elements as a list, whose orders are preserved.
    """
    def small():
        ret = []
        for e in iterable:
            if e not in ret:
                ret.append(e)
        return ret

    def large():
        ret = []
        memo = set()
        for e in iterable:
            if e not in memo:
                memo.add(e)
                ret.append(e)
        return ret

    if hasattr(iterable, '__len__'):
        if len(iterable) < 1000:
            return small()
    return large()


def where(iterable):
    """Get the positions where conditions are not False.

    Parameters
    ----------
    iterable : collections.Iterable[any]
        The conditions to be checked.

    Returns
    -------
    list[int]
        Positions where the conditions are not False.
    """
    return [i for i, c in enumerate(iterable) if c]


def round_int(value):
    """Cast the specified value to nearest integer."""
    if isinstance(value, float):
        return int(round(value))
    return int(value)


def yield_parent_dirs(path, include_self=False):
    """
    Traverse through all the parent directories.

    Parameters
    ----------
    path : str
        Starting path.

    include_self : bool
        Whether or not to include the starting path itself.
    """
    parent, child = os.path.split(path)
    if include_self:
        yield path
    while path != parent:
        if child and child != '.':
            yield parent
        path = parent
        parent, child = os.path.split(path)


def flatten_list(root):
    """Flatten the given list.

    All the non-list elements would be aggregated to the root-level,
    with the same order as they appear in the original list.

    Parameters
    ----------
    root : collections.Iterable[any]
        The list to be flatten.

    Returns
    -------
    list[any]
    """
    ret = []
    try:
        stack = list(reversed(root))
    except TypeError:
        stack = [root]
    while stack:
        u = stack.pop()
        if isinstance(u, list):
            for v in reversed(u):
                stack.append(v)
        else:
            ret.append(u)
    return ret


def check_argtype(arg, types, arg_name=None):
    """Check whether ``arg`` is instance of ``types``.

    Parameters
    ----------
    arg : any
        Argument value.

    types : type | tuple[type]
        Type or tuple of types, just as used in instanceof.

    arg_name : str
        If specified, will be used to construct the error message.

    Raises
    ------
    TypeError
        If ``arg`` is not instance of ``types``.
    """
    if not isinstance(arg, types):
        if arg_name:
            raise TypeError('Argument %r is expected to be %r, but got %r.' %
                            (arg_name, types, arg))
        else:
            raise TypeError('Argument is expected to be %r, but got %r.' %
                            (types, arg))


def write_string(file, text, encoding='utf-8'):
    """Write text to given file.

    This method will automatically detect whether or not ``file``
    is text based or byte based.
    """
    if not isinstance(text, six.string_types):
        raise TypeError('Text must be string or bytes type.')
    if isinstance(file, io.TextIOBase):
        if not isinstance(text, six.text_type):
            text = text.decode(encoding)
        return file.write(text)
    else:
        if not isinstance(text, six.binary_type):
            text = text.encode(encoding)
        return file.write(text)


def assert_raises_message(test_case, error_type, message):
    class _AssertRaisesMessageContext(object):
        def __init__(self, owner, ctx):
            self.owner = owner
            self.ctx = ctx

        def __enter__(self):
            self.ctx.__enter__()

        def __exit__(self, exc_type, exc_val, exc_tb):
            ret = self.ctx.__exit__(exc_type, exc_val, exc_tb)
            self.owner.assertEquals(str(self.ctx.exception), message)
            return ret

    return _AssertRaisesMessageContext(
        test_case, test_case.assertRaises(error_type))


@contextlib.contextmanager
def safe_overwrite_file(filename, temp_suffix='.swap'):
    """Get a context to overwrite file safely.

    This method will make a context to overwrite specified file safely.
    It achieves this by telling the caller to write to a temporary file,
    instead of to the target file, then move the temporary file to actual
    target if everything goes fine.

    Parameters
    ----------
    filename : str
        The target file to be overwritten.

    temp_suffix : str
        Suffix for the temporary file.
        The temporary file will be created under the same directory as target.
    """
    temp_file = None
    try:
        # ensure the parent directory exists
        parent_dir = os.path.dirname(os.path.abspath(filename))
        if not os.path.isdir(parent_dir):
            os.makedirs(parent_dir)

        # search for a target file which does not exist
        temp_file = '%s%s' % (filename, temp_suffix)
        exist_counter = 0
        while os.path.exists(temp_file):
            temp_file = '%s%s.%s' % (filename, temp_suffix, uuid.uuid4())
            exist_counter += 1
            if exist_counter > 1000:
                raise RuntimeError('Failed to get a temporary file name.')

        # yield this temporary file
        yield temp_file

        # everything is okay, now move the temp file to target file
        # be careful, we must make sure the rename is atomic
        atomic_replace(temp_file, filename)
    finally:
        # delete temp_file if it exists
        if temp_file is not None and os.path.exists(temp_file):
            os.remove(temp_file)


def atomic_replace(source, destination):
    """Use `source` file to resplace `destination` in atomic operation."""
    try:
        os.replace(source, destination)     # Python 3.3 and better.
    except AttributeError:
        if sys.platform == 'win32':
            # FIXME This is definitely not atomic!
            # But it's how (for example) Mercurial does it, as of 2016-03-23
            # https://selenic.com/repo/hg/file/tip/mercurial/windows.py
            try:
                os.rename(source, destination)
            except OSError as err:
                if err.winerr != 183:
                    raise
                os.remove(destination)
                os.rename(source, destination)
        else:
            # Atomic on POSIX. Not sure about Cygwin, OS/2 or others.
            os.rename(source, destination)


def ensure_dir_exists(path):
    """Ensure the specified directory exists, even under concurrent issues.

    This method will try to create the directory for up to 3 times, and raise
    error if all the attempts fail.  It will sleep for a random time, up to
    10ms, between each failure.
    """
    attempt = 0
    while not os.path.isdir(path):
        try:
            os.makedirs(path)
            break
        except IOError:
            attempt += 1
            if attempt >= 3:
                raise
        # sleep for a random time, up to 10ms, so as to avoid races.
        time.sleep(random.random() / 100.0)
    return path


def ensure_parent_exists(path):
    """Ensure the parent directory of specified path exists."""
    parent = os.path.split(path)[0]
    if parent:
        ensure_dir_exists(os.path.abspath(parent))
    return path


def get_process_identity():
    """Get a unique identity for current process.

    This method will generate a unique identity for current process,
    regarding the host and the process id.
    """
    import os
    import socket
    pid = os.getpid()
    host = socket.gethostname()
    return '%s/%s' % (host, pid)


@contextlib.contextmanager
def fileno_redirected(src_fd, dst_fd):
    """Temporarily redirect outputs for one file descriptor to another.

    Parameters
    ----------
    src_fd : int
        Source file descriptor which should be redirected.

    dst_fd : int
        Target file descriptor where to redirect output into.
    """
    dup_fd = os.dup(src_fd)
    try:
        os.dup2(dst_fd, src_fd)
        yield
    finally:
        os.dup2(dup_fd, src_fd)


@contextlib.contextmanager
def file_redirected(original_file, redirected_file):
    """Temporarily redirect outputs for one file to another.

    The two files must be both system files, i.e., having file descriptor in
    the system, thus file-like objects are not supported.

    Parameters
    ----------
    original_file : io.IOBase
        Original file which should be redirected.

    redirected_file : io.IOBase | str
        Target file where to redirect output to, or the path to target file.
    """
    if isinstance(redirected_file, six.string_types):
        target_file = open(redirected_file, 'wb')
    else:
        target_file = None
    try:
        dup_fileno = None
        try:
            dst = target_file or original_file
            dup_fileno = os.dup(dst.fileno())
            os.dup2(dst.fileno(), original_file.fileno())
            yield
        finally:
            if dup_fileno is not None:
                os.dup2(dup_fileno, original_file.fileno())
                os.close(dup_fileno)
    finally:
        if target_file is not None:
            target_file.close()


def init_logging():
    """Initialize logging according to Madoka config."""
    import logging.config
    from madoka import config
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': config.log_format
            },
        },
        'handlers': {
            'default': {
                'level': config.log_level,
                'formatter': 'default',
                'class': 'logging.StreamHandler',
            },
        },
        'loggers': {
            '': {
                'handlers': ['default'],
                'level': config.log_level,
                'propagate': True
            },
            'tensorflow': {
                'handlers': ['default'],
                'level': 'WARN',
                'propagate': True
            },
        }
    })


def wrap_text_writer(out, encoding, manage=True):
    """Wrap a binary output stream as a text writer.

    Parameters
    ----------
    out : io.IOBase
        The binary output stream.

    encoding : str
        The encoding of the text writer.

    manage : bool
        If set to True, will close the underlying binary stream when
        the text writer has been closed.
    """
    if isinstance(out, io.TextIOBase):
        #
        return out
    elif isinstance(out, io.RawIOBase):
        buffer = io.BufferedIOBase(out)
        if not manage:
            buffer.close = lambda: None
    else:
        # This is to handle passed objects that aren't in the
        # IOBase hierarchy, but just have a write method
        buffer = io.BufferedIOBase()
        buffer.writable = lambda: True
        buffer.write = out.write
        if manage:
            buffer.close = out.close
        try:
            # TextIOWrapper uses this methods to determine
            # if BOM (for UTF-16, etc) should be added
            buffer.seekable = out.seekable
            buffer.tell = out.tell
        except AttributeError:
            pass

    # wrap a binary writer with TextIOWrapper
    class UnbufferedTextIOWrapper(io.TextIOWrapper):
        def write(self, s):
            super(UnbufferedTextIOWrapper, self).write(s)
            self.flush()
    return UnbufferedTextIOWrapper(buffer, encoding=encoding,
                                   errors='xmlcharrefreplace',
                                   newline='\n')


def is_mask_array(arr):
    """Whether or not the specified array is a mask array?"""
    return arr.dtype == np.bool


@contextlib.contextmanager
def set_current_directory(path):
    """Open a context with specified current directory."""
    curdir = os.path.abspath(os.path.curdir)
    os.chdir(os.path.abspath(path))
    try:
        yield
    finally:
        os.chdir(curdir)


def filter_not_none(iterable):
    """Filter out None values in given iterable collection.

    Parameters
    ----------
    iterable : collections.Iterable
        The iterable collection to be filtered.

    Returns
    -------
    list
        Returns the list of non-None elements.
    """
    return [i for i in iterable if i is not None]


class AutoJsonEncoder(json.JSONEncoder):
    """
    JSONEncoder subclass that leverages an object's `__json__()` method,
    if available, to obtain its default JSON representation.
    """

    def default(self, obj):
        try:
            return obj.__json__()
        except AttributeError:
            return json.JSONEncoder.default(self, obj)


@contextlib.contextmanager
def duplicate_console_output(path, stderr=False, append=False):
    """Copy the STDOUT and STDERR to both the console and a file.

    Parameters
    ----------
    path : str
        Path of the output file.

    stderr : bool
        The console output, including STDOUT and STDERR, would be aggregated,
        thus there will only be one console output within the open context.

        If set to False, will aggregate both STDOUT and STDERR into STDOUT.
        Otherwise will aggregate both STDOUT and STDERR into STDERR.

    append : bool
        Whether or not to open the output file in append mode?
    """
    # flush the original stdout and stderr
    sys.stdout.flush()
    sys.stderr.flush()

    # determine the arguments of calling `_tee.py`
    args = [
        sys.executable,
        '-u',
        os.path.abspath(os.path.join(os.path.dirname(__file__), '_tee.py')),
        '--file',
        os.path.abspath(path),
    ]
    if stderr:
        args.append('--stderr')
    if append:
        args.append('--append')

    # open the subprocess and get its stdin file descriptor
    proc = subprocess.Popen(args, stdin=subprocess.PIPE)
    proc_fd = proc.stdin.fileno()

    # now redirect the STDOUT and STDERR
    try:
        with fileno_redirected(sys.stdout.fileno(), proc_fd), \
                fileno_redirected(sys.stderr.fileno(), proc_fd):
            yield
    finally:
        os.close(proc_fd)
        proc.wait()
