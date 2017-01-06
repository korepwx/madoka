# -*- coding: utf-8 -*-
import codecs
import os
import time
from logging import getLogger
from stat import S_ISDIR

from madoka.utils import TrainStorage, NamedStatusCode

__all__ = [
    'TrainStorageStatus', 'TrainStorageDir', 'TrainStorageSnapshot',
    'TrainStorageScanner'
]


class TrainStorageStatus(NamedStatusCode):
    """Training storage statuses."""

    # The training is still running
    RUNNING = 0

    # The training has done, with reports generated
    DONE = 1

    # The training has done, but reports have not been generated
    STOPPED = 2


class TrainStorageDir(object):
    """Training storage directory.

    Parameters
    ----------
    path : str
        Virtual path of the directory.

    tags : collections.Iterable[str]
        List of tags associated with this storage.

    status : int
        The status of this training storage, one of the codes in
        `TrainStorageStatus`.

    modify_time : float
        The timestamp of the last modify time.
    """

    def __init__(self, path, tags=None, status=None, modify_time=None):
        if tags is not None:
            tags = list(tags)
        self.path = path
        self.tags = tags
        self.status = status
        self.modify_time = modify_time

    def __eq__(self, other):
        return isinstance(other, TrainStorageDir) and (
            (self.path, self.tags, self.status, self.modify_time) ==
            (other.path, other.tags, other.status, other.modify_time)
        )

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        status = TrainStorageStatus.get_name(self.status) or self.status
        return 'TrainStorageDir(%r,%r,%r,%r)' % (
            self.path, self.tags, status, self.modify_time)

    def __json__(self):
        return {
            'path': self.path,
            'tags': self.tags,
            'status': self.status,
            'modify_time': self.modify_time
        }

    @property
    def is_running(self):
        return self.status == TrainStorageStatus.RUNNING


def _format_list(obj, depth=1):
    indent = ' ' * (depth * 2)
    sep = ',\n' + indent
    ret = sep.join(repr(x) for x in obj)
    if ret:
        ret = '[\n%s%s\n%s]' % (indent, ret, ' ' * (depth * 2 - 2))
    else:
        ret = '[]'
    return ret


class TrainStorageSnapshot(object):
    """Snapshot of training storage.

    Parameters
    ----------
    storage_dirs : list[TrainingStorageDir]
        List of storage directories instances.
    """

    def __init__(self, storage_dirs):
        storage_dirs = sorted(storage_dirs, key=lambda x: x.path)
        self._storage_dirs = storage_dirs   # type: list[TrainStorageDir]

    def __iter__(self):
        return iter(self._storage_dirs)

    def __len__(self):
        return len(self._storage_dirs)

    def __getitem__(self, item):
        return self._storage_dirs[item]

    def __repr__(self):
        if self._storage_dirs:
            ret = 'TrainStorageSnapshot(%s)' % _format_list(self._storage_dirs)
        else:
            ret = 'TrainStorageSnapshot()'
        return ret

    def __json__(self):
        return self._storage_dirs

    def compute_diff(self, old):
        """Compute the difference of this snapshot upon the old one.

        Parameters
        ----------
        old : TrainStorageSnapshot
            The old snapshot.

        Returns
        -------
        TrainStorageSnapshotDiff
        """
        left = 0
        right = 0
        created = []
        updated = []
        deleted = []

        while left < len(old) and right < len(self):
            if old[left].path < self[right].path:
                deleted.append(old[left])
                left += 1
            elif old[left].path > self[right].path:
                created.append(self[right].path)
                right += 1
            else:
                if old[left] != self[right]:
                    updated.append(self[right])
                left += 1
                right += 1

        if left < len(old):
            deleted.extend(old[left:])
        if right < len(self):
            created.extend(self[right:])

        return TrainStorageSnapshotDiff(created, updated, deleted)


class TrainStorageSnapshotDiff(object):
    """Difference of training storage snapshot.

    Parameters
    ----------
    created : list[TrainingStorageDir]
        Newly created training storage directories.

    updated : list[TrainingStorageDir]
        Updated training storage directories.

    deleted : list[TrainingStorageDir]
        Deleted training storage directories.
    """

    def __init__(self, created, updated, deleted):
        self.created = created
        self.updated = updated
        self.deleted = deleted

    def __repr__(self):
        elements = []
        for k in ('created', 'updated', 'deleted'):
            v = getattr(self, k)
            if v:
                elements.append('%s=%s,' % (k, _format_list(v, 2)))
        if not elements:
            ret = 'TrainStorageSnapshotDiff()'
        else:
            ret = 'TrainStorageSnapshotDiff(\n  %s\n)' % '\n  '.join(elements)
        return ret

    def __len__(self):
        return len(self.created) + len(self.updated) + len(self.deleted)


class TrainStorageScanner(object):
    """Scanner of training storage directories.

    This scanner will probably cache the old scanning results, in order
    to speedup later scans.

    Parameters
    ----------
    root_path : str
        Root path of the training storage directory.

        The `TrainStorageDir` instances created by this scanner will have
        relative path as virtual path (according to this root path).

    scan_depth : int
        Maximum depth of recursive scanning.
        If not specified, will literally have no limit.

    mtime_depth : int
        Maximum depth for counting last modify-time.
        If not specified, will only count the modify-time for two levels.
    """

    def __init__(self, root_path, scan_depth=(1 << 30), mtime_depth=2):
        self.root_path = os.path.abspath(root_path)
        self.scan_depth = scan_depth
        self.mtime_depth = mtime_depth

        # store the result of last scan.
        self._last_scan = {}    # type: dict[str, TrainStorageDir]

    def scan(self):
        """Scan the training storage directories.

        Returns
        -------
        TrainStorageSnapshot
            A snapshot of the training storage directories.
        """
        def join_path(a, b):
            if a:
                return a + '/' + b
            else:
                return b

        def scan_dir(target, fs_path, virtual_path, scan_depth):
            versions_file = os.path.join(fs_path, TrainStorage._VERSION_FN)
            scan_exclude = set()
            if os.path.isfile(versions_file):
                storage_dict = TrainStorage.list(fs_path, lock_versions=False)
                for name in storage_dict:
                    tags = sorted(storage_dict[name])
                    storage = TrainStorageDir(join_path(virtual_path, name),
                                              tags=tags)
                    target.append(storage)
                    # add the storage name and its tags to exclude set
                    # so that we will not scan these entities afterwards
                    scan_exclude.add(name)
                    for t in tags:
                        scan_exclude.add(t)
            if scan_depth > 0:
                scan_depth -= 1
                for f in os.listdir(fs_path):
                    if f not in scan_exclude:
                        path = os.path.join(fs_path, f)
                        if os.path.isdir(path):
                            scan_dir(target, path, join_path(virtual_path, f),
                                     scan_depth)
            # we hope this will benefit the gc collector
            scan_exclude.clear()

        def read_running_flag(path):
            running_flag = os.path.join(path, TrainStorage._RUNNING_FN)
            if os.path.exists(running_flag):
                with codecs.open(running_flag, 'rb', 'utf-8') as f:
                    cnt = f.read()
                    lines = [x for x in cnt.split('\n') if x]
                    try:
                        if len(lines) != 3:
                            raise ValueError('Number of lines %r != 3.' %
                                             len(lines))
                        pid, host, atime = tuple(lines)
                        atime = float(atime)
                        atime_diff = time.time() - atime
                        # twice the interval of dumping running flag
                        interval = TrainStorage._RUNNING_FLAG_DUMP_INTERVAL
                        return atime_diff < interval * 2
                    except ValueError:
                        getLogger(__name__).warning(
                            'Unrecognized running flag file %r:\n%s',
                            path, cnt, exc_info=1
                        )
            return False

        def read_modify_time(path, mtime_depth):
            try:
                st = os.stat(path)
                mtime = st.st_mtime
                if S_ISDIR(st.st_mode) and mtime_depth > 0:
                    mtime_depth -= 1
                    for f in os.listdir(path):
                        mtime = max(
                            mtime,
                            read_modify_time(os.path.join(path, f),
                                             mtime_depth)
                        )
                return mtime
            except (IOError, OSError):
                return 0

        # first, gather all storage directories
        storages = []   # type: list[TrainStorageDir]
        scan_dir(storages, self.root_path, '', self.scan_depth)

        # second, fill storage information
        for s in storages:
            s_path = os.path.join(self.root_path, s.path)

            # read the status
            is_running = read_running_flag(s_path)
            if is_running:
                s.status = TrainStorageStatus.RUNNING
            else:
                report_file = os.path.join(s_path,
                                           TrainStorage._REPORT_DIR,
                                           TrainStorage._REPORT_HTML_FN)
                if os.path.exists(report_file):
                    s.status = TrainStorageStatus.DONE
                else:
                    s.status = TrainStorageStatus.STOPPED

            # We can take the modify time from last scan if it existed
            # in the last scan, it was not running in the last scan and
            # it is not running currently.
            last_s = self._last_scan.get(s.path, None)
            if last_s and not last_s.is_running and s.status == last_s.status:
                s.modify_time = last_s.modify_time
            else:
                s.modify_time = read_modify_time(s_path, self.mtime_depth)

        # memorize the results of this scan
        self._last_scan = {s.path: s for s in storages}

        # construct the snapshot
        return TrainStorageSnapshot(storages)
