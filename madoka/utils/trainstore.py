# -*- coding: utf-8 -*-
import codecs
import contextlib
import gzip
import os
import pickle as pkl
import random
import shutil
import socket
import sys
import threading
import time
import traceback
from datetime import datetime
from logging import getLogger

import six

from .misc import (ensure_dir_exists, ensure_parent_exists,
                   duplicate_console_output)
from .pathlock import PathLock

__all__ = ['TrainStorage']


def _symlink(source, link_name, target_is_directory=False):
    if six.PY3:
        kwargs = {'target_is_directory': target_is_directory}
    else:
        kwargs = {}
    return os.symlink(source, link_name, **kwargs)


class _RunningFlag(object):
    """The `running` file to indicate the activity of current process.

    This class will rewrite the running file, located at specified path,
    every few seconds, so as to indicate that the current process is
    still active.
    """

    def __init__(self, path, interval):
        self.path = os.path.abspath(path)
        self.interval = interval
        self._stopped = False
        self._worker = None         # type: threading.Thread
        self._start_sem = None      # type: threading.Semaphore
        self._stop_sem = None       # type: threading.Semaphore

    def _dump(self):
        cnt = '\n'.join([
            str(os.getpid()),
            socket.gethostname(),
            str(time.time())
        ])
        with codecs.open(self.path, 'wb', 'utf-8') as f:
            f.write(cnt)

    def _thread_run(self):
        # notify the `open()` method that we've successfully started.
        self._start_sem.release()

        # dump the running flag every per `interval` until stopped.
        while not self._stopped:
            try:
                self._dump()
            except Exception:
                getLogger(__name__).warning(
                    'Failed to write running flag.', exc_info=1)

            # sleep by using the condition, so that we can interrupt
            # it gracefully.
            if self._stop_sem.acquire(timeout=self.interval):
                break

    def open(self):
        # initialize synchronization objects
        self._stopped = False
        self._start_sem = threading.Semaphore(0)
        self._stop_sem = threading.Semaphore(0)
        # create the worker thread
        self._worker = threading.Thread(target=self._thread_run)
        self._worker.daemon = True
        self._worker.start()
        # wait for the worker to actually start
        self._start_sem.acquire()

    def close(self):
        # notify the worker thread to exit
        self._stopped = True
        self._stop_sem.release()

        # wait for the worker thread to exit
        self._worker.join()
        self._worker = None
        try:
            if os.path.exists(self.path):
                os.remove(self.path)
        except Exception:
            getLogger(__name__).warning(
                'Failed to remove running flag.', exc_info=1)


class TrainStorage(object):
    """Storage for TensorFlow training data.

    When training TensorFlow models, it is a common practice to write
    summaries and save checkpoints on disk.  However, it brings extra
    trouble for managing training data of different runs, generated
    by the same program.

    This class helps maintain a directory of all historical training
    data generated by the same program.

    A training context can be open via `with` statement.  When leaving
    such context, the checkpoint directory inside the storage will be
    deleted if everything goes okay, or some error has taken place but
    `keep_failure_checkpoint` is not set to True.
    """

    _PICKLE_PROTOCOL = 3  # for Python 2.7 and Python 3.x
    _NO_LIMIT = 1 << 30
    _RUNNING_FLAG_DUMP_INTERVAL = 120

    _LATEST_FN = 'latest'
    _VERSION_FN = 'version'
    _RUNNING_FN = 'running'

    _CHECKPOINT_DIR = 'checkpoints'
    _SUMMARY_DIR = 'summary'
    _PERSIST_DIR = 'persist'
    _REPORT_DIR = 'report'
    _SESSOIN_PERSIST_FN = 'session.ckpt'
    _REPORT_PICKLE_FN = 'report.pkl.gz'
    _REPORT_HTML_FN = 'report.html'
    _MAIN_SCRIPT_FN = 'main.py'

    _MKDIR_LOCK_FN = 'mkdir.lock'
    _VERSION_LOCK_FN = 'version.lock'

    _LOGGING_FN = 'logging.txt'

    @classmethod
    def _check_latest_link(cls, root_dir):
        latest_file = os.path.join(root_dir, cls._LATEST_FN)
        if os.path.exists(latest_file) and not os.path.islink(latest_file):
            raise IOError('%r exists but is not a link.' % latest_file)
        return latest_file

    @classmethod
    def _mklock(cls, path, timeout=10):
        from madoka import config
        path = os.path.abspath(path)
        if config.train_lock_zk:
            ret = PathLock.zookeeper(
                config.train_lock_zk, path, timeout=timeout)
        else:
            ret = PathLock.plain(path, timeout=timeout)
        return ret

    @classmethod
    def _lock_make_dir(cls, root_dir):
        return cls._mklock(os.path.join(root_dir, cls._MKDIR_LOCK_FN))

    @classmethod
    def _read_versions(cls, root_dir):
        version_file = os.path.join(root_dir, cls._VERSION_FN)
        versions = []
        if os.path.exists(version_file):
            with codecs.open(version_file, 'rb', 'utf-8') as f:
                versions = list(filter(
                    lambda x: x,
                    (s.strip() for s in f)
                ))
        return versions

    @classmethod
    def _lock_versions(cls, root_dir, max_versions=_NO_LIMIT):
        """Lock the versions file and open a context to modify the versions."""
        latest_file = cls._check_latest_link(root_dir)

        class VersionEditor(object):
            def __init__(self, versions):
                self.versions = list(versions)
                self._original_version = list(self.versions)
                self.purged_versions = []
                self.removed_versions = []

            @property
            def has_changed(self):
                return not not (
                    self._original_version != self.versions or
                    self.purged_versions or
                    self.removed_versions
                )

            def check_purge(self):
                all_versions = self.versions + self.purged_versions
                self.versions = all_versions[: max_versions]
                self.purged_versions = all_versions[max_versions:]

            def add(self, version):
                self.versions.insert(0, version)
                self.check_purge()

            def remove(self, version):
                self.versions.remove(version)
                self.removed_versions.append(version)
                self.check_purge()

        @contextlib.contextmanager
        def versions_context():
            with cls._mklock(os.path.join(root_dir, cls._VERSION_LOCK_FN)):
                # read versions list
                version_file = os.path.join(root_dir, cls._VERSION_FN)
                old_versions = cls._read_versions(root_dir)

                # yield the versions editor for context
                editor = VersionEditor(old_versions)
                yield editor

                # save the new versions list
                if editor.has_changed:
                    with codecs.open(version_file, 'wb', 'utf-8') as f:
                        f.write('\n'.join(editor.versions))
                    # purge old directories
                    for p in editor.purged_versions + editor.removed_versions:
                        path = os.path.join(root_dir, p)
                        try:
                            shutil.rmtree(path)
                        except Exception:
                            getLogger(__name__).warn(
                                'Failed to purge training directory %r.',
                                path,
                                exc_info=1
                            )
                            # ignore the error of purging old directory,
                            # since it is not so severe to keep it on disk.

                    # Create symbol link to refer to the latest work directory
                    # We do not create the latest link on Windows, since it
                    # requires special privilege to create soft links.
                    if sys.platform != 'win32':
                        if os.path.islink(latest_file):
                            os.unlink(latest_file)
                        if editor.versions:
                            work_dir = os.path.join(os.path.abspath(root_dir),
                                                    editor.versions[0])
                            _symlink(
                                os.path.relpath(
                                    work_dir, os.path.dirname(latest_file)),
                                latest_file,
                                target_is_directory=True
                            )

        return versions_context()

    @classmethod
    def create(cls, root_dir, max_versions=_NO_LIMIT, **kwargs):
        """Create a new training directory.

        This method will create a new training directory under `root_dir`,
        update the "latest" symbolic link to refer this new directory,
        and purge old training directories if required.

        Parameters
        ----------
        root_dir : str
            Root directory that contains the training directories.

        max_versions : int
            Keep at most this number of historical training directories.
            Old training directories will be purged automatically.
            If not specified, will keep all of the old directories.

        **kwargs
            Additional argument passed to TrainStorage constructor.

        Returns
        -------
        TrainStorage
            A train storage instance of the created training directory.
        """
        root_dir = os.path.abspath(root_dir)
        cls._check_latest_link(root_dir)

        # pick up a free working directory
        def make_name():
            return datetime.strftime(datetime.now(), '%Y%m%d.%H%M%S.%f')[:-3]

        def make_work_dir():
            name = make_name()
            return name, os.path.join(root_dir, name)

        ensure_dir_exists(root_dir)
        with cls._lock_make_dir(root_dir):
            work_name, work_dir = make_work_dir()
            attempt = 0
            while os.path.exists(work_dir) and attempt < 10:
                time.sleep(random.random() / 100.0)  # sleep for up to 10ms
                work_name, work_dir = make_work_dir()
                attempt += 1
            if attempt >= 10:
                raise IOError('Failed to choose a unique working directory.')
            ensure_dir_exists(work_dir)

        # Commit the working directory to versions file
        with cls._lock_versions(root_dir, max_versions) as versions:
            versions.add(work_name)

        # construct the storage instance
        getLogger(__name__).info(
            'Training directory %r has been created.' % work_dir)
        return TrainStorage(work_dir, **kwargs)

    @classmethod
    def open(cls, root_dir, name=_LATEST_FN, require_checkpoints=True,
             **kwargs):
        """Open a training directory to recover from previous training.

        Parameters
        ----------
        root_dir : str
            Root directory that contains the training directories.

        name : str
            Name or tag of the required training directory (default "latest").

        require_checkpoints : bool
            If set to True, will refuse to open the training directory
            if checkpoints do not exist.

        **kwargs
            Additional argument passed to TrainStorage constructor.

        Returns
        -------
        TrainStorage
            A train storage instance of the opened training directory.

        Raises
        ------
        IOError
            If `require_checkpoints` is True and the checkpoints do not exist.
        """
        with cls._lock_make_dir(root_dir):
            work_dir = os.path.abspath(os.path.join(root_dir, name))
            if not os.path.isdir(work_dir):
                raise IOError('Training directory %s/%s does not exist.' %
                              (root_dir, name))
            checkpoint_dir = os.path.join(work_dir, cls._CHECKPOINT_DIR)
            if require_checkpoints and not os.path.isdir(checkpoint_dir):
                raise IOError('No checkpoints under training directory '
                              '%s/%s.' % (root_dir, name))
            # It is very important to get the real path instead of just
            # the absolute path of specified work_dir, so that even if the
            # symbolic link of original path changes, it will not affect
            # a created TrainStorage instance.
            getLogger(__name__).info(
                'Training directory %r has been opened.' % work_dir)
            return TrainStorage(os.path.realpath(work_dir), **kwargs)

    @classmethod
    def list(cls, root_dir, lock_versions=True):
        """List all training directories under given storage root.

        Parameters
        ----------
        root_dir : str
            Root directory that contains the training directories.

        lock_versions : bool
            Whether or not to lock the versions file.

        Returns
        -------
        dict[str, list[str]]
            A dict that contains name of the training directories
            as its keys, and list of tags (symbolic link names) as
            its values, corresponding to the keys.
        """
        root_dir = os.path.realpath(root_dir)

        # lock the versions file if required
        if lock_versions:
            with cls._lock_versions(root_dir, cls._NO_LIMIT):
                return cls.list(root_dir, lock_versions=False)

        # list the directory with out locking the versions
        ret = {k: [] for k in cls._read_versions(root_dir)}

        # gather all the links
        for f in os.listdir(root_dir):
            path = os.path.join(root_dir, f)
            if os.path.islink(path):
                for k in ret:
                    dst = os.path.join(root_dir, k)
                    if os.path.samefile(dst, path):
                        ret[k].append(f)
                        break
        return ret

    @classmethod
    def delete(cls, root_dir, name):
        """Delete an existing training directory.

        All the symbol links that points to this directory will also be
        removed.  Furthermore, the "latest" link will be re-created if
        necessary.

        Parameters
        ----------
        root_dir : str
            Root directory that contains the training directories.

        name : str
            Name or tag of the required training directory.
        """
        work_dir = os.path.realpath(os.path.join(root_dir, name))
        if not os.path.isdir(work_dir):
            raise IOError('Training directory %s/%s does not exist.' %
                          (root_dir, name))

        # gather all the links.
        links = []
        for f in os.listdir(root_dir):
            # latest link will be treated by `_lock_versions`, thus
            # should not be modified here.
            if f != cls._LATEST_FN:
                path = os.path.join(root_dir, f)
                if os.path.islink(path):
                    if os.path.samefile(work_dir, os.path.realpath(path)):
                        links.append(path)

        # modify the versions file
        name = os.path.split(work_dir)[1]
        with cls._lock_versions(root_dir, cls._NO_LIMIT) as versions:
            versions.remove(name)

        # finally, delete all links
        for link in links:
            try:
                os.unlink(link)
            except Exception:
                getLogger(__name__).warning(
                    'Failed to delete link %r.', link, exc_info=1)

        # log this deletion
        getLogger(__name__).info(
            'Training directory %r has been deleted.' % work_dir)

    @classmethod
    def add_tags(cls, root_dir, name, tags):
        """Add tags to an existing training directory.

        Will throw exception if any of the specified tags already exists,
        and does not point to the given training directory.

        Parameters
        ----------
        root_dir : str
            Root directory that contains the training directories.

        name : str
            Name or tag of the required training directory.

        tags : collections.Iterable[str]
            List of tags to be added.
        """
        root_dir = os.path.abspath(root_dir)
        target_dir = os.path.realpath(os.path.join(root_dir, name))
        if not os.path.isdir(target_dir):
            raise IOError('Training directory %s/%s does not exist.' %
                          (root_dir, name))

        with cls._lock_versions(root_dir):
            # check the links
            new_tags = []
            for t in tags:
                path = os.path.join(root_dir, t)
                if os.path.exists(path):
                    if not os.path.samefile(target_dir, path):
                        raise ValueError(
                            'Tag %r exists but does not point to %r.' %
                            (t, name)
                        )
                else:
                    new_tags.append(t)

            # add the tags
            for t in new_tags:
                _symlink(
                    os.path.relpath(target_dir, root_dir),
                    os.path.join(root_dir, t),
                    target_is_directory=True
                )

    @classmethod
    def remove_tags(cls, root_dir, name, tags):
        """Remove tags from an existing training directory.

        Will throw exception if any of the specified tags exists, but
        does not point to the given training directory.

        Parameters
        ----------
        root_dir : str
            Root directory that contains the training directories.

        name : str
            Name or tag of the required training directory.

        tags : collections.Iterable[str]
            List of tags to be added.
        """
        root_dir = os.path.abspath(root_dir)
        target_dir = os.path.realpath(os.path.join(root_dir, name))
        if not os.path.isdir(target_dir):
            raise IOError('Training directory %s/%s does not exist.' %
                          (root_dir, name))

        with cls._lock_versions(root_dir):
            # check the links
            delete_tags = []
            for t in tags:
                path = os.path.join(root_dir, t)
                if t == cls._LATEST_FN:
                    raise ValueError('The latest tag cannot be removed.')
                elif os.path.exists(path):
                    if not os.path.samefile(target_dir, path):
                        raise ValueError(
                            'Tag %r exists but does not point to %r.' %
                            (t, name)
                        )
                    delete_tags.append(t)
                else:
                    raise IOError('Tag %r does not exist.' % t)

            # delete the tags
            for t in delete_tags:
                os.unlink(os.path.join(root_dir, t))

    def __init__(self,
                 work_dir,
                 capture_logging=True,
                 logging_format=None,
                 purge_checkpoint=False,
                 keep_failure_checkpoint=False):
        """Construct a new TraingStorage instance.

        Parameters
        ----------
        work_dir : str
            The path of the training directory.

        capture_logging : bool
            Whether or not to capture the STDOUT, STDERR and logged messages
            during a training context?

        logging_format : str
            The captured logging format.
            If not specified, will use ``madoka.config.log_format``.

        purge_checkpoint : bool
            Whether or not to purge the checkpoint files after training?

        keep_failure_checkpoint : bool
            Whether or not to keep the checkpoint files when training failed?

            If True, will not purge the checkpoints after a failed training,
            even if `purge_checkpoint` is set to True.
        """
        from madoka import config
        if logging_format is None:
            logging_format = config.log_format

        # memorize arguments
        self.work_dir = os.path.abspath(work_dir)
        self.capture_logging = capture_logging
        self.logging_format = logging_format
        self.purge_checkpoint = purge_checkpoint
        self.keep_failure_checkpoint = keep_failure_checkpoint

        # initialize the working directory
        ensure_dir_exists(self.resolve_path(self._CHECKPOINT_DIR))
        ensure_dir_exists(self.resolve_path(self._SUMMARY_DIR))
        ensure_dir_exists(self.resolve_path(self._PERSIST_DIR))
        ensure_dir_exists(self.resolve_path(self._REPORT_DIR))

        # create the running flag object
        self._running_flag = _RunningFlag(
            self.ensure_parent_exists(self._RUNNING_FN),
            self._RUNNING_FLAG_DUMP_INTERVAL
        )

        # Lazy initialized members.
        self._console_output_context = None

    @property
    def name(self):
        """Get the name of the training directory."""
        return os.path.split(self.work_dir)[1]

    def resolve_path(self, *paths):
        """Join pieces of paths and make it absolute relative to work_dir.

        Parameters
        ----------
        *paths : tuple[str]
            Path pieces relative to the work_dir of this storage.

        Returns
        -------
        str
            Resolved absolute path.
        """
        return os.path.abspath(os.path.join(self.work_dir, *paths))

    def ensure_parent_exists(self, *paths):
        """Resolve the path pieces and ensure its parent exists."""
        path = self.resolve_path(*paths)
        ensure_parent_exists(path)
        return path

    @property
    def checkpoint_dir(self):
        return self.resolve_path(self._CHECKPOINT_DIR)

    @property
    def summary_dir(self):
        return self.resolve_path(self._SUMMARY_DIR)

    @property
    def persist_dir(self):
        return self.resolve_path(self._PERSIST_DIR)

    @property
    def report_dir(self):
        return self.resolve_path(self._REPORT_DIR)

    def save_session(self, filename=_SESSOIN_PERSIST_FN):
        """Save the current session as a checkpoint file.

        Parameters
        ----------
        filename : str
            The name of the checkpoint file, relative to the persist directory.
        """
        import tensorflow as tf
        from madoka.utils.tfhelper import ensure_default_session
        sess = ensure_default_session()
        saver = tf.train.Saver()
        path = self.ensure_parent_exists(self._PERSIST_DIR, filename)
        saver.save(sess, path)

    def restore_session(self, filename=_SESSOIN_PERSIST_FN):
        """Restore the current session from a checkpoint file.

        Parameters
        ----------
        filename : str
            The name of the checkpoint file, relative to the persist directory.
        """
        import tensorflow as tf
        from madoka.utils.tfhelper import ensure_default_session
        sess = ensure_default_session()
        saver = tf.train.Saver()
        path = os.path.join(self._PERSIST_DIR, filename)
        saver.restore(sess, path)

    def save_object(self, obj, filename):
        """Save an object as file.

        Parameters
        ----------
        obj : any
            Object to be saved.

        filename : str
            The name of the file, relative to the persist directory.
        """
        persist_file = self.ensure_parent_exists(self._PERSIST_DIR, filename)
        with gzip.open(persist_file, 'wb') as f:
            pkl.dump(obj, f, protocol=self._PICKLE_PROTOCOL)

    def load_object(self, filename):
        """Load an object from file.

        Parameters
        ----------
        filename : str
            The name of the file, relative to the persist directory.

        Returns
        -------
        any
            The loaded object.
        """
        with gzip.open(os.path.join(self._PERSIST_DIR, filename), 'rb') as f:
            return pkl.load(f)

    def save_report(self, report_or_reports, exp_id=None,
                    pickle_filename=_REPORT_PICKLE_FN,
                    html_filename=_REPORT_HTML_FN):
        """Save an evaluation report.

        Parameters
        ----------
        report_or_reports : Report | collection.Iterable[Report]
            Report or a list of reports.

        exp_id : str
            Identifier of the experiment.
            If not specified, will use the path of this training storage.

        pickle_filename : str
            File name of the report pickle file.

        html_filename : str
            File name of the report HTML file.
        """
        # check the reports
        from madoka.report import (Report, ReportGroup, TFSummaryReport,
                                   TFSummaryReportGroup, tf_summary_report)

        def is_summary_report(r):
            return isinstance(r, (TFSummaryReport, TFSummaryReportGroup))

        if isinstance(report_or_reports, ReportGroup) \
                and report_or_reports.name is None:
            reports = report_or_reports.children
        elif isinstance(report_or_reports, Report):
            reports = [report_or_reports]
        else:
            reports = list(report_or_reports)
        if not any(is_summary_report(r) for r in reports) and \
                len(os.listdir(self.summary_dir)) > 0:
            reports.append(tf_summary_report(self.summary_dir))
        if len(reports) == 1:
            report = reports[0]
        else:
            report = ReportGroup(reports)

        # check the experiment information
        if exp_id is None:
            exp_id = self.work_dir

        # create directories
        pickle_path = self.ensure_parent_exists(self._REPORT_DIR,
                                                pickle_filename)
        html_path = self.ensure_parent_exists(self._REPORT_DIR,
                                              html_filename)

        # save the original report object
        with gzip.open(pickle_path, 'wb') as f:
            pkl.dump(report, f, protocol=self._PICKLE_PROTOCOL)

        # render the report html page
        from madoka.report.renderer import HTMLReportRenderer
        with HTMLReportRenderer.open(html_path, exp_id=exp_id) as renderer:
            renderer.render(report)

    def load_report(self, pickle_filename=_REPORT_PICKLE_FN):
        """Load an evaluation report.

        Parameters
        ----------
        pickle_filename : str
            File name of the report pickle file.

        Returns
        -------
        Report
            The report object.
        """
        from madoka.report import Report
        pickle_path = self.resolve_path(self._REPORT_DIR, pickle_filename)
        with gzip.open(pickle_path, 'rb') as f:
            ret = pkl.load(f)
        if not isinstance(ret, Report):
            raise TypeError('Object loaded but is not a report.')
        return ret

    def save_script(self, script_path, save_filename=_MAIN_SCRIPT_FN):
        """Save a script file to the storage.

        Parameters
        ----------
        script_path : str
            Path of the script file.  You may use the `__file__` magic
            variable to get the running script path.

        save_filename : str
            File name saved to the storage.
        """
        dest_path = self.ensure_parent_exists(save_filename)
        shutil.copy(script_path, dest_path)

    def __enter__(self):
        """Start a training context."""
        # duplicate all the outputs to both console and log file
        log_file = self.resolve_path(self._LOGGING_FN)
        self._console_output_context = \
            duplicate_console_output(log_file, append=True)
        self._console_output_context.__enter__()

        # setup the running flag of this process
        self._running_flag.open()

        # use this object itself as context object
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # delete the running flag
        self._running_flag.close()

        # delete the checkpoint files
        if self.purge_checkpoint and \
                not (exc_type and self.keep_failure_checkpoint):
            chk_path = self.resolve_path(self._CHECKPOINT_DIR)
            try:
                if os.path.exists(chk_path):
                    shutil.rmtree(chk_path)
            except Exception:
                getLogger(__name__).warning(
                    'Failed to cleanup checkpoint directory %r.' % chk_path,
                    exc_info=1
                )
        # if error occurred, log it
        if exc_type is not None:
            err = traceback.format_exception(exc_type, exc_val, exc_tb)
            getLogger(__name__).warning(
                'Closing training storage with error.\n%s',
                ''.join(err)
            )
        else:
            getLogger(__name__).debug('Closing training storage with success.')

        # restore console output
        if self._console_output_context:
            self._console_output_context.__exit__(exc_type, exc_val, exc_tb)
            self._console_output_context = None
