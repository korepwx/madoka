# -*- coding: utf-8 -*-
import codecs
import hashlib
import json
import os
import string
import sys

import six

from madoka.utils import safe_overwrite_file, ensure_parent_exists

__all__ = ['ReportResourcesManager']


class _HashDB(object):
    """Simple database for tracking stored resources.

    Parameters
    ----------
    path : str
        Path of the database.
    """

    def __init__(self, path):
        self.path = path
        self._items = {}
        if os.path.exists(self.path):
            self.load()

    def load(self):
        with codecs.open(self.path, 'rb', 'utf-8') as f:
            self._items = json.load(f)

    def save(self):
        ensure_parent_exists(self.path)
        with safe_overwrite_file(self.path) as path:
            with codecs.open(path, 'wb', 'utf-8') as f:
                json.dump(self._items, f)

    @staticmethod
    def _join_hashes(hashes):
        if not isinstance(hashes, six.string_types):
            hashes = ';'.join(hashes)
        return hashes

    @staticmethod
    def _split_hashes(hashes):
        if isinstance(hashes, six.string_types):
            hashes = hashes.split(';')
        return hashes

    def get(self, hashes):
        """Get the name of files whose content matches given hash codes.

        Parameters
        ----------
        hashes : str | tuple[str]
            Hash code, or a tuple of hash codes.

        Returns
        -------
        list[str]
        """
        return self._items.get(self._join_hashes(hashes), [])

    def add(self, hashes, filename):
        """Add the name of a file to the database, with specified hash codes.

        Parameters
        ----------
        hashes : str | tuple[str]
            Hash code, or a tuple of hash codes.

        filename : str
            Name of the file.
        """
        k = self._join_hashes(hashes)
        if k not in self._items:
            self._items[k] = []
        self._items[k].append(filename)
        self.save()


class ReportResourcesManager(object):
    """Manages the report resource files.

    A report usually contains various resources, e.g., images and CSV tables.
    Such resources may be referred in report files via hyper-links.
    However, it is often the case that multiple types of reports are to be
    generated, which shares a large set of resource files, but generated
    by different renderer instances.

    In order for these renders to actually share the resource files, we have
    this class to manage the generated resources.  Any resources with same
    content and type will be saved as a same file.  Files with same content
    but different types will be saved as different symbolic links that point
    to the same file if possible, or different individual files if the system
    does not support symbolic links.

    Parameters
    ----------
    root_dir : str
        Path of the resources directory.
    """

    _DB_FN = 'resources.db'
    _NONAME_FN = 'noname'

    _FILENAME_DIGITS = string.ascii_lowercase + string.digits
    _FILENAME_DIGITS_NUM = len(_FILENAME_DIGITS)
    _GET_FREE_FILENAME_MAX_RETRY = 10000

    def __init__(self, root_dir):
        # check the resources directory
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)
        root_dir = os.path.abspath(root_dir)
        self.root_dir = root_dir

        # open the hash database
        self.hash_db = _HashDB(self.resolve_path(self._DB_FN))

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        """Open the resource manager for operations."""

    def close(self):
        """Write all changes and close the resources manager."""

    def resolve_path(self, *pieces):
        """Join path pieces and resolve under the root directory."""
        return os.path.join(self.root_dir, *pieces)

    def _ensure_parent_exists(self, *pieces):
        """Resolve the path, and ensure its parent directory exists."""
        path = self.resolve_path(*pieces)
        ensure_parent_exists(path)
        return path

    @staticmethod
    def _compute_hash(data):
        return (
            hex(len(data)),
            hashlib.md5(data).hexdigest(),
            hashlib.sha1(data).hexdigest(),
        )

    @staticmethod
    def _split_fullext(filename):
        extensions = []
        root, ext = os.path.splitext(filename)
        while ext:
            extensions.append(ext)
            root, ext = os.path.splitext(root)
        return root, ''.join(reversed(extensions))

    def _get_free_filename(self, filename, exist_files=None):
        if not os.path.exists(self.resolve_path(filename)):
            return filename
        fn, ext = self._split_fullext(filename)
        exist_files = set(exist_files or ())
        for i in range(2, self._GET_FREE_FILENAME_MAX_RETRY):
            filename2 = '%s_%d%s' % (fn, i, ext)
            if filename2 in exist_files:
                return filename2
            if not os.path.exists(self.resolve_path(filename2)):
                return filename2
        raise IOError('Cannot find a free filename for %r.' % filename)

    def _make_filename(self, hashes, extension, exist_files=None):
        hashcode = int(hashes[-1], 16)
        hashstr = []
        while hashcode > 0:
            hashstr.append(
                self._FILENAME_DIGITS[hashcode % self._FILENAME_DIGITS_NUM])
            hashcode //= 36
        hashstr = ''.join(reversed(hashstr))
        return self._get_free_filename(
            '%s/%s%s' % (self._NONAME_FN, hashstr, extension),
            exist_files
        )

    def _save(self, data, ext, filename):
        # check the arguments.
        if ext is None and filename is None:
            raise TypeError('At least one of `extension` and `filename` '
                            'should be specified.')
        if filename is not None:
            ext = self._split_fullext(filename)
        elif not ext.startswith('.'):
            raise ValueError('%r is not a file extension.' % ext)

        # compute the file hash
        hashes = self._compute_hash(data)
        exist_files = self.hash_db.get(hashes)

        # if the suggested file name is not specified, we can reuse any
        # existing file with the same extension.
        if filename is None:
            for f in exist_files:
                if f.endswith(ext):
                    return f
            filename = self._make_filename(hashes, ext, exist_files)

        # Otherwise we verify whether or not the existing file is same
        # as what we are going to save.
        # If so, return the existing file name, or otherwise we should
        # find a new one.
        elif os.path.exists(self.resolve_path(filename)):
            if filename in exist_files:
                return filename
            filename = self._get_free_filename(filename, exist_files)
            # Note that `filename` might be occupied by other file content,
            # but there might be some other file name with the same content
            # in `exist_files`.
            if filename in exist_files:
                return filename

        # now we save the file, or make a symbolic link to existing file
        dstfile = self._ensure_parent_exists(filename)
        with safe_overwrite_file(dstfile) as tmpfile:
            if exist_files and sys.platform != 'win32':
                source_file = self.resolve_path(exist_files[0])
                link_parent = os.path.split(dstfile)[0]
                os.symlink(os.path.relpath(source_file, link_parent), tmpfile)
            else:
                with open(tmpfile, 'wb') as f:
                    f.write(data)

            # finally, save this file name in the database, and return it.
            self.hash_db.add(hashes, filename)
        return filename

    def save_bytes(self, data, extension=None, filename=None):
        """Save bytes as a resource file.

        Parameters
        ----------
        data : bytes
            Resource content in bytes.

        extension : str
            File extension of this resource.
            Will be ignored if `filename` is specified.

        filename : str
            Suggested file name of this resource.

            If the desired filename has been occupied, will find a new
            but similar filename.

        Returns
        -------
        The file name of this resource.
        """
        return self._save(data, extension, filename)

    def save_file(self, file, extension=None, filename=None):
        """Save bytes as a resource file.

        Parameters
        ----------
        file : str | io.IOBase
            Path to the file, or a file-like object, which should be saved.

        extension : str
            File extension of this resource.
            Will be ignored if `filename` is specified.

        filename : str
            Suggested file name of this resource.

            If the desired filename has been occupied, will find a new
            but similar filename.

        Returns
        -------
        The file name of this resource.
        """
        if isinstance(file, six.string_types):
            if extension is None:
                extension = self._split_fullext(file)[1]
            with open(file, 'rb') as f:
                return self._save(f.read(), extension, filename)
        elif hasattr(file, 'read'):
            if extension is None and hasattr(file, 'name'):
                extension = self._split_fullext(file.name)[1]
            return self._save(file.read(), extension, filename)
        else:
            raise TypeError('`file` is required to be a path or a file object.')
