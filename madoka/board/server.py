# -*- coding: utf-8 -*-
import contextlib
import copy
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from logging import getLogger
from stat import S_ISDIR
from subprocess import CalledProcessError

import six
from flask import Blueprint, Flask, render_template, current_app, request
from flask.helpers import safe_join
from gevent.pywsgi import WSGIServer
from pygments import highlight
from pygments.formatters.html import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from werkzeug.exceptions import NotFound, BadRequest

from madoka.utils import TrainStorage
from .snapshot import (TrainStorageSnapshot, TrainStorageSnapshotDiff,
                       TrainStorageScanner)

__all__ = ['MadokaBoardServer']


class _SnapshotCollector(object):
    """Background thread to collect training storage snapshots.

    Parameters
    ----------
    storage_root : dict[str, str]
        Path of the training storage, or mappings from URL prefix to
        paths of training storage.

    scan_interval : int
        Number of seconds between two successive scans of training storage.
    """

    def __init__(self, storage_root, scan_interval):
        self.storage_root = storage_root
        self.scan_interval = scan_interval
        self.scanners = {
            k: TrainStorageScanner(v)
            for k, v in six.iteritems(storage_root)
        }
        self.snapshot = None        # type: TrainStorageSnapshot

        # the background watcher
        self._watching = False
        self._watcher = None        # type: threading.Thread
        self._start_sem = None      # type: threading.Semaphore
        self._sleep_sem = None      # type: threading.Semaphore

        # initialize the snapshot for the first time
        self.collect()

    def collect(self):
        """Collect the snapshot and difference.

        Returns
        -------
        TrainStorageSnapshotDiff
            The difference between old and new snapshots.
        """
        storage_dirs = []
        for prefix, scanner in six.iteritems(self.scanners):
            for s in scanner.scan():
                s = copy.copy(s)
                s.path = prefix + '/' + s.path
                storage_dirs.append(s)
        snapshot = TrainStorageSnapshot(storage_dirs)
        if self.snapshot is not None:
            snapshot_diff = snapshot.compute_diff(self.snapshot)
        else:
            snapshot_diff = None
        self.snapshot = snapshot
        return snapshot_diff

    def wakeup_watcher(self):
        """Wake up the background thread immediately."""
        if self._watching:
            self._sleep_sem.release()

    def start_watcher(self):
        """Start the background thread to watch training storage."""
        def watch():
            self._start_sem.release()
            while self._watching:
                try:
                    start_time = time.time()
                    diff = self.collect()
                    stop_time = time.time()
                    seconds = stop_time - start_time

                    getLogger(__name__).info(
                        'Collected snapshot in %.2f seconds, %s update(s).',
                        seconds, len(diff)
                    )
                    if getLogger(__name__).isEnabledFor(logging.DEBUG):
                        getLogger(__name__).debug('%s', diff)
                except Exception:
                    getLogger(__name__).warning(
                        'Failed to collect snapshot.', exc_info=1)
                self._sleep_sem.acquire(timeout=self.scan_interval)

        self._watching = True
        self._start_sem = threading.Semaphore(0)
        self._sleep_sem = threading.Semaphore(0)
        self._watcher = threading.Thread(target=watch, daemon=True)
        self._watcher.start()
        # wait for the thread to actually startup.
        self._start_sem.acquire()

    def stop_watcher(self):
        """Stop the background watcher."""
        if self._watching:
            self._watching = False
            self._sleep_sem.release()


def json_response(method):
    """Decorator to make json response."""
    @six.wraps(method)
    def inner(*args, **kwargs):
        return (
            json.dumps(method(*args, **kwargs)),
            200,
            {'Content-Type': 'application/json'}
        )
    return inner


_STORAGE_META_FILES = [
    TrainStorage._LATEST_FN, TrainStorage._MKDIR_LOCK_FN,
    TrainStorage._VERSION_FN, TrainStorage._VERSION_LOCK_FN,
]


def _chown_to_parent(parent, check_list=None, ignore_errors=False):
    """
    Set the owner of all storage directories and tags to that of the
    parent directory.

    Parameters
    ----------
    parent : str
        The parent directory of storage and tags.

    check_list : collections.Iterable[str]
        If specified, will only check these entries under the parent.
        If not, will check all entries recursively.

    ignore_errors : bool
        Whether or not to ignore errors of chown?
    """
    p_st = os.stat(parent)

    # function to check whether or not a given path should be chown
    def should_chown(file, st=None):
        if st is None:
            st = os.stat(file)
        if st.st_uid != p_st.st_uid or st.st_gid != p_st.st_gid:
            return True

    # gather the entity list which should be chown
    if check_list is None:
        def gather_list(path, dst):
            for f in os.listdir(path):
                f_path = os.path.join(path, f)
                f_st = os.stat(f_path)
                if should_chown(f_path, f_st):
                    dst.append(f_path)
                if S_ISDIR(f_st.st_mode):
                    gather_list(f_path, dst)
            return dst
    else:
        def gather_list(path, dst):
            for f in check_list:
                f_path = os.path.join(path, f)
                if should_chown(f_path):
                    dst.append(f_path)
            return dst

    try:
        file_list = gather_list(parent, [])

        # chown the entities
        for file in file_list:
            try:
                os.chown(file, p_st.st_uid, p_st.st_gid)
            except Exception:
                if not ignore_errors:
                    raise
                getLogger(__name__).warning(
                    'Failed to change owner of %r to %s:%s.',
                    file, p_st.st_uid, p_st.st_gid, exc_info=1
                )
    except Exception:
        if not ignore_errors:
            raise
        getLogger(__name__).warning(
            'Failed to gather entities under %r.', parent, exc_info=1)


def _make_report(parent, name, exp_id, fix_permissions=False):
    # actually generate the report.
    with TrainStorage.open(
            parent, name, require_checkpoints=False) as store:
        report = store.load_report()
        store.save_report(report, exp_id=exp_id)

    # fix the permissions
    if fix_permissions:
        report_dir = TrainStorage._REPORT_DIR
        _chown_to_parent(
            parent,
            _STORAGE_META_FILES + [report_dir],
            ignore_errors=True
        )
        _chown_to_parent(
            os.path.join(parent, name, report_dir),
            ignore_errors=True
        )


class _StorageBlueprint(Blueprint):
    """Flask blueprint to serve training reports under specified root.

    Parameters
    ----------
    name : str
        Name of this blue print.

    root_path : str
        Root path of the training storage.
    """

    def __init__(self, name, root_path, url_prefix):
        super(_StorageBlueprint, self).__init__(
            name, __name__, url_prefix=url_prefix
        )
        self.root_path = os.path.realpath(root_path)

        # add url routes
        self.add_url_rule('/<path:path>/_navigation.html',
                          view_func=self.navigation)
        self.add_url_rule('/<path:path>/_main.py',
                          view_func=self.main_script)
        self.add_url_rule('/<path:path>/_api',
                          view_func=self.storage_api,
                          methods=['POST'])

        # auto index generator
        from flask_autoindex import AutoIndexBlueprint
        self._index_generator = AutoIndexBlueprint(
            self,
            browse_root=self.root_path
        )

    def get_storage_root(self, path):
        """Get storage root for given virtual path.

        Parameters
        ----------
        path : str
            The virtual path of the storage, relative to `self.root_path`.

        Returns
        -------
        str
            The file system path of the storage.

        Raises
        ------
        NotFound
            If the specified storage does not exist.
        """
        # get the storage root path
        path = [p for p in re.sub(r'(/|\\)+', '/', path).split('/')
                if p not in ('', '.', '..')]
        root = os.path.join(self.root_path, *path)

        # check whether or not it looks like a storage root directory
        if not os.path.isdir(root):
            raise NotFound()
        summary_dir = os.path.join(root, TrainStorage._SUMMARY_DIR)
        report_dir = os.path.join(root, TrainStorage._REPORT_DIR)
        logging_file = os.path.join(root, TrainStorage._LOGGING_FN)
        candidates = (summary_dir, report_dir, logging_file)
        if not any(os.path.exists(p) for p in candidates):
            raise NotFound()

        return root

    def navigation(self, path):
        self.get_storage_root(path)     # ensure the storage exists
        exp_id = path.rsplit('/', 1)[-1]
        return render_template('navigation.html', exp_id=exp_id)

    def main_script(self, path):
        # ensure the storage exists
        root = self.get_storage_root(path)

        # read the main script
        script_file = os.path.join(root, 'main.py')
        if not os.path.exists(script_file):
            raise NotFound()
        with open(script_file, 'rb') as f:
            code = f.read()

        # highlight the code by pygments
        lexer = get_lexer_by_name("python", stripall=True)
        formatter = HtmlFormatter(linenos=True, cssclass="codehilite")
        source = highlight(code, lexer, formatter)
        return render_template(
            'highlight.html', source=source, filename='main.py')

    def _storage_delete(self, path, parent, name):
        try:
            TrainStorage.delete(parent, name)
            if current_app.server.chown_to_parent:
                _chown_to_parent(
                    parent, _STORAGE_META_FILES, ignore_errors=True)
            current_app.server.collector.wakeup_watcher()
            return {'error': 0, 'message': ''}
        except Exception:
            msg = 'Failed to delete storage %r.' % path
            getLogger(__name__).warning(msg, exc_info=1)
            return {'error': 1, 'message': msg}

    def _storage_change_tags(self, path, parent, name, tags, op):
        try:
            if op == 'remove_tags':
                TrainStorage.remove_tags(parent, name, tags)
            elif op == 'add_tags':
                TrainStorage.add_tags(parent, name, tags)
            if current_app.server.chown_to_parent:
                _chown_to_parent(
                    parent, _STORAGE_META_FILES + tags, ignore_errors=True)
            current_app.server.collector.wakeup_watcher()
            return {'error': 0, 'message': ''}
        except ValueError as ex:
            msg = str(ex)
            return {'error': 1, 'message': msg}
        except Exception:
            msg = 'Failed to %s tags %r for storage %r.' % (op, tags, path)
            getLogger(__name__).warning(msg, exc_info=1)
            return {'error': 1, 'message': msg}

    def _storage_make_report(self, path, parent, name):
        fix_permissions = current_app.server.chown_to_parent
        if self.url_prefix.endswith('/'):
            exp_id = self.url_prefix + path
        else:
            exp_id = self.url_prefix + '/' + path

        # spawn the python process to make the report
        import gevent.subprocess
        try:
            env = copy.copy(os.environ)
            env['PYTHONPATH'] = '%s:%s' % (
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__),
                        '../..'
                    )
                ),
                env.get('PYTHONPATH', '')
            )
            _ = gevent.subprocess.check_output(
                [
                    sys.executable,
                    '-c',
                    '# -*- encoding: utf-8 -*-\n'
                    'from madoka.board.server import _make_report\n'
                    '_make_report(%r, %r, %r, %r)' %
                    (parent, name, exp_id, fix_permissions)
                ],
                stderr=gevent.subprocess.STDOUT
            )
            return {'error': 0, 'message': ''}
        except CalledProcessError as ex:
            msg = (
                'Failed to generate report for storage %r: '
                'exit code is %d.' % (path, ex.returncode)
            )
            getLogger(__name__).warning(msg)
            getLogger(__name__).info(ex.output.decode('utf-8'))
            return {'error': 1, 'message': msg}

    @json_response
    def storage_api(self, path):
        # parse the request body
        action = request.get_json(force=True)
        if not isinstance(action, dict):
            raise BadRequest()
        op = action['op']
        if op not in ('delete', 'make_report', 'add_tags', 'remove_tags'):
            raise BadRequest()

        # ensure the storage exists
        root = self.get_storage_root(path)
        parent, name = os.path.split(root.rstrip('/'))
        if not parent:
            msg = 'Parent directory of storage %r is empty!' % path
            return {'error': 1, 'message': msg}

        if op == 'delete':
            # dispatch the delete action
            ret = self._storage_delete(path, parent, name)
        elif op == 'make_report':
            # dispatch the make report action
            ret = self._storage_make_report(path, parent, name)
        else:
            # parse tags
            if 'tags' not in action:
                raise BadRequest()
            tags = action['tags']
            if not isinstance(tags, list) or \
                    not all(isinstance(s, str) for s in tags):
                raise BadRequest()

            # dispatch the tags action
            ret = self._storage_change_tags(path, parent, name, tags, op)
        return ret


class _RootApp(Flask):
    """Flask application of Madoka Board.

    Parameters
    ----------
    server : MadokaBoardServer
        The Madoka Board application server.
    """

    def __init__(self, server):
        super(_RootApp, self).__init__(__name__, static_url_path='/_static')

        # memorize the server
        self.server = server

        # add url routes
        self.add_url_rule('/', view_func=self.index)
        self.add_url_rule('/_settings', view_func=self.settings)
        self.add_url_rule('/_snapshot', view_func=self.snapshot)

        # cache of static file hashes
        self._file_hash_cache = {}

    def index(self):
        return render_template('index.html')

    @json_response
    def settings(self):
        return {
            'scan_interval': self.server.scan_interval
        }

    @json_response
    def snapshot(self):
        return [
            (s.path, s.tags, s.status, s.modify_time)
            for s in self.server.collector.snapshot
        ]

    def inject_url_defaults(self, endpoint, values):
        """
        Injects an "h" parameter on the URLs of static files that contains a
        hash of the file.  This allows the use of aggressive cache settings on
        static files, while ensuring that content changes are reflected
        immediately due to the changed URLs.  Hashes are cached in-memory
        and only checked for updates when the file mtime changes.

        Source: https://gist.github.com/mfenniak/2978805
        """
        super(_RootApp, self).inject_url_defaults(endpoint, values)
        if endpoint == "static" and "filename" in values:
            filepath = safe_join(self.static_folder, values["filename"])
            if os.path.isfile(filepath):
                cache = self._file_hash_cache.get(filepath)
                mtime = os.path.getmtime(filepath)
                if cache != None:
                    cached_mtime, cached_hash = cache
                    if cached_mtime == mtime:
                        values["h"] = cached_hash
                        return
                h = hashlib.md5()
                with contextlib.closing(open(filepath, "rb")) as f:
                    h.update(f.read())
                h = h.hexdigest()
                self._file_hash_cache[filepath] = (mtime, h)
                values["h"] = h


class MadokaBoardServer(object):
    """Madoka Board server.

    Parameters
    ----------
    storage_root : str | dict[str, str]
        Path of the training storage, or mappings from URL prefix to
        paths of training storage.

    scan_interval : int
        Number of seconds between two successive scans of training storage.
        Default is 120.

    port : int
        Bind port number.  Default is 8080.

    interface : str
        Bind interface.
        If not specified, will bind to all interfaces.

    chown_to_parent : bool
        Whether or not to set the owner of newly created tags to
        that of the parent directory?

    debug : bool
        Whether or not to turn on Flask debugging features?
        Default is False.
    """

    def __init__(self, storage_root, scan_interval=120, port=8080,
                 interface=None, chown_to_parent=False, debug=False):
        # memorize arguments
        if not isinstance(storage_root, dict):
            storage_root = {'': os.path.realpath(storage_root)}
        else:
            storage_root = {
                k.rstrip('/'): os.path.realpath(v)
                for k, v in six.iteritems(storage_root)
            }
        self.storage_root = storage_root
        self.scan_interval = scan_interval
        self.port = port
        self.interface = interface or ''
        self.chown_to_parent = chown_to_parent

        # create the snapshot collector
        self.collector = _SnapshotCollector(storage_root, scan_interval)

        # construct Flask application
        self.app = _RootApp(self)
        for k, v in six.iteritems(storage_root):
            bp_name = 'storage_bp_%s' % v.strip('/').replace('/', '_')
            self.app.register_blueprint(
                _StorageBlueprint(bp_name, v, url_prefix=k)
            )

        # turn on debug features
        if debug:
            self.app.config['DEBUG'] = True
            self.app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

        # make the Flask application aware of template changes
        self.app.config['TEMPLATES_AUTO_RELOAD'] = True

        # initialize the server
        self.http = WSGIServer((self.interface, self.port), self.app)

    def run(self):
        """Run the server in foreground."""
        if self.interface:
            endpoint = '%s:%s' % (self.interface, self.port)
        else:
            endpoint = str(self.port)
        getLogger(__name__).info(
            'Madoka Board server listening at %s', endpoint)

        self.collector.start_watcher()
        self.http.serve_forever()
