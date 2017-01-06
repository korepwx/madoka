# -*- coding: utf-8 -*-
import os
import re
import unittest

import sys

from madoka.report import ReportResourcesManager
from madoka.utils import TemporaryDirectory, set_current_directory


class ReportRendererUnitTest(unittest.TestCase):
    """Unit tests for report renderers."""

    def test_resources_manager(self):
        """Test report resource manager."""
        with TemporaryDirectory() as tmpdir:
            # content-based checks
            with ReportResourcesManager(tmpdir) as rrm:
                resources = (
                    # (filename, extension, data)
                    ('1.txt', None, b'123'),
                    ('1.bat', None, b'1234'),
                    (None, '.txt', b'1234'),
                    (None, '.bat', b'1234'),
                    ('1.txt', None, b'123'),
                    ('1.txt', None, b'1234'),
                    ('1.txt', None, b'12345'),
                    ('1.txt', None, b'12345'),
                    ('nested/1.bat', None, b'123'),
                )
                filenames = []

                for fn, ext, data in resources:
                    filename = rrm.save_bytes(data, extension=ext, filename=fn)
                    filenames.append(filename)
                    if fn is not None:
                        fn, ext = os.path.splitext(fn)
                        self.assertTrue(
                            filename.startswith(fn),
                            msg='%r is expected to start with %r.' %
                                (filename, fn)
                        )
                    self.assertEquals(ext, os.path.splitext(filename)[1])

                for i, (_, _, data) in enumerate(resources):
                    filename = filenames[i]
                    with open(os.path.join(tmpdir, filename), 'rb') as f:
                        cnt = f.read()
                        self.assertEquals(
                            cnt,
                            data,
                            msg='content of %r is expected to be %r,'
                                ' but got %r.' % (filename, data, cnt)
                        )

            # filename-based checks
            with ReportResourcesManager(tmpdir) as rrm, \
                    set_current_directory(tmpdir):
                self.assertEquals(
                    rrm.save_bytes(b'123', filename='1.txt'), '1.txt')
                self.assertEquals(
                    rrm.save_bytes(b'123', filename='123.txt'), '123.txt')
                if sys.platform != 'win32':
                    self.assertTrue(os.path.islink('123.txt'))
                    self.assertTrue(os.path.samefile('1.txt', '123.txt'))
                self.assertEquals(
                    rrm.save_bytes(b'1234', filename='1.txt'), '1_2.txt')
                self.assertEquals(
                    rrm.save_bytes(b'123', filename='nested/1.bat'),
                    'nested/1.bat'
                )
                self.assertEquals(
                    rrm.save_bytes(b'1234', filename='nested/1.bat'),
                    'nested/1_2.bat'
                )
                self.assertEquals(
                    rrm.save_bytes(b'123', extension='.txt'), '1.txt')
                self.assertEquals(
                    rrm.save_bytes(b'123', extension='.bat'), 'nested/1.bat')
                jpg_file = rrm.save_bytes(b'123', extension='.jpg')
                self.assertTrue(
                    re.match(r'^noname\/.*\.jpg$', jpg_file))
                self.assertEquals(
                    rrm.save_bytes(b'123', extension='.jpg'), jpg_file)
