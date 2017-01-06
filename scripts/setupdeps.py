#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script to install dependencies of this project."""

import codecs
import os
import re
import subprocess
import sys

script_root = os.path.abspath(os.path.split(__file__)[0])

# detect the environment
platform = {
    'linux2': 'linux',
    'linux': 'linux',
    'darwin': 'mac',
    'win32': 'windows',
}.get(sys.platform, None)

if not platform:
    print('Platform %s not supported.' % sys.platform)
    sys.exit(1)

try:
    ANACONDA_SIGNATURE = (
        'Anaconda',
        'Continuum',
    )
    subprocess.check_call(['conda', '--version'])
    is_anaconda = any(s in sys.version for s in ANACONDA_SIGNATURE)
except Exception:
    is_anaconda = False

py_ver = '.'.join(str(s) for s in sys.version_info[:2])
if py_ver not in ('2.7', '3.4', '3.5'):
    print('Python version %s not supported.' % py_ver)
    sys.exit(1)

if 'TF_DEVICE' in os.environ:
    device = os.environ['TF_DEVICE']
else:
    try:
        subprocess.check_call(['nvcc', '--version'])
        device = 'gpu'
    except Exception:
        device = 'cpu'


# Read dependencies from requirements.txt
tf_deps = []  # which will be installed after TensorFlow is installed
conda_deps = []
pip_deps = []
active_deps = pip_deps
requirements_file = os.path.join(script_root, '../requirements.txt')

with codecs.open(requirements_file, 'rb', 'utf-8') as f:
    for line in f:
        line = line.strip()
        if line.startswith('# TensorFlow'):
            active_deps = tf_deps
        elif line.startswith('# Anaconda'):
            active_deps = conda_deps
        elif line.startswith('# Pip'):
            active_deps = pip_deps
        elif line and not line.startswith('#'):
            # there might be whitespaces in the line, and we want to
            # throw away all these characters.
            active_deps.append(''.join(line.split()))

# If the python is Anaconda based, install some packages via conda
if conda_deps:
    if is_anaconda:
        subprocess.check_call(['conda', 'install', '-y'] + conda_deps)
    else:
        pip_deps = conda_deps + pip_deps

# Install dependencies via pip
if pip_deps:
    subprocess.check_call(['pip', 'install'] + pip_deps)

# Install TensorFlow
tf_packages = {
    ('linux', '2.7'): 'tensorflow%(pkgsuffix)s-%(ver)s-cp27-none-linux_x86_64',
    ('linux', '3.4'): 'tensorflow%(pkgsuffix)s-%(ver)s-cp34-cp34m-linux_x86_64',
    ('linux', '3.5'): 'tensorflow%(pkgsuffix)s-%(ver)s-cp35-cp35m-linux_x86_64',
    ('mac', '2.7'): 'tensorflow%(pkgsuffix)s-%(ver)s-py2-none-any',
    ('mac', '3.4'): 'tensorflow%(pkgsuffix)s-%(ver)s-py3-none-any',
    ('mac', '3.5'): 'tensorflow%(pkgsuffix)s-%(ver)s-py3-none-any',
    ('windows', '3.5'): 'tensorflow%(pkgsuffix)s-%(ver)s-cp35-cp35m-win_amd64',
}
tf_versions = {
    ('linux', '2.7'): '0.12.0rc1',
    ('linux', '3.4'): '0.12.0rc1',
    ('linux', '3.5'): '0.12.0rc1',
    ('mac', '2.7'): '0.12.0rc1',
    ('mac', '3.4'): '0.12.0rc1',
    ('mac', '3.5'): '0.12.0rc1',
    ('windows', '3.5'): '0.12.0rc1',
}
if (platform, py_ver) in tf_packages:
    tf_pkg = tf_packages[(platform, py_ver)]
    tf_ver = tf_versions[(platform, py_ver)]
else:
    raise ValueError(
        'Platform %r is not supported.' % ((platform, py_ver, device),))
tf_url = (
    'https://storage.googleapis.com/tensorflow/%(platform)s/%(device)s/' +
    tf_pkg + '.whl'
)
tf_url %= {
    'platform': platform,
    'device': device,
    'py_ver': py_ver,
    'ver': tf_ver,
    'pkgsuffix': '' if device == 'cpu' else '_gpu'
}
subprocess.check_call(['pip', 'install', tf_url])

# Install dependencies after TensorFlow
if tf_deps:
    tf_deps = list(filter(
        lambda s: not re.match(r'tensorflow', s, re.I),
        tf_deps
    ))
    subprocess.check_call(['pip', 'install'] + tf_deps)
