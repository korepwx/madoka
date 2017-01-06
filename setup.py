#!/usr/bin/env python
import ast
import codecs
import re

from setuptools import setup, find_packages

_version_re = re.compile(r'__version__\s+=\s+(.*)')

with codecs.open('madoka/__init__.py', 'rb', 'utf-8') as f:
    version = str(ast.literal_eval(_version_re.search(f.read()).group(1)))

with codecs.open('requirements.txt', 'rb', 'utf-8') as f:
    install_requires = [line for line in f.readlines()
                        if line.strip() and not line.startswith('#')]

setup(
    name='madoka',
    version=version,
    description='Madoka ML Toolkit.',
    author='Haowen Xu',
    author_email='public@korepwx.com',
    url='https://git.peidan.me/madoka/madoka',
    packages=find_packages('madoka'),
    platforms='any',
    setup_requires=['setuptools'],
    install_requires=install_requires
)
