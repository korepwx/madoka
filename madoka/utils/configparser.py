# -*- coding: utf-8 -*-
import os
import sys
import warnings

import six

from . import misc
from .configattr import ConfigAttr

__all__ = [
    'ConfigParser', 'MergeConfigParser', 'PyFileConfigParser',
    'EnvVarConfigParser', 'DefaultConfigParser',
]


class ConfigParser(object):
    """Base class for loading config values from external sources."""

    def _gather_attr_list(self, owner):
        ret = []
        meet_attr = {}
        for member_name in dir(owner.__class__):
            attr = getattr(owner.__class__, member_name)
            if isinstance(attr, ConfigAttr):
                attr_name = attr.name.lower()
                if attr_name in meet_attr and \
                        meet_attr[attr_name] is not attr:
                    raise KeyError('Duplicated config attribute %r.' %
                                   attr.name)
                ret.append((member_name, attr))
                meet_attr[attr_name] = attr
        return ret

    def _load_attr_values(self, attr_dict):
        """Load values of attributes according to attribute dict.

        Derived classes should implement this to actual load values.
        If any of the requested attribute does not exist in external source,
        it might be excluded from the returning dict.

        Parameters
        ----------
        attr_dict : dict[str, ConfigAttr]
            Dict of attributes, from their name to the attribute instances,
            which should be loaded.  The attribute names are guaranteed to
            be lower-cased.

        Returns
        -------
        dict[str, any]
            Values for these attributes, as dict.
        """
        raise NotImplementedError()

    def load(self, owner):
        """Load all config attributes of ``owner``.

        Derived classes should override ``_load_attr_values`` instead of
        this method.

        Parameters
        ----------
        owner : any
            Any object which contains ``ConfigAttr`` instances.
        """
        attr_list = self._gather_attr_list(owner)
        name_to_member = {a.name.lower(): k for k, a in attr_list}
        attr_dict = {a.name.lower(): a for k, a in attr_list}
        attr_values = self._load_attr_values(attr_dict)
        for attr_name, value in six.iteritems(attr_values):
            if attr_name in name_to_member:
                setattr(owner, name_to_member[attr_name], value)


class MergeConfigParser(ConfigParser):
    """Config parser that merges values from multiple parsers.

    This class will load values from specified parsers one by one.
    Values from later parsers will thus overwrite the former ones.

    Parameters
    ----------
    parsers : collections.Iterable[ConfigParser]
        Underlying parsers for this merge parser.
    """

    def __init__(self, parsers):
        self.parsers = list(parsers)

    def _load_attr_values(self, attr_dict):
        attr_values = {}
        for p in self.parsers:
            attr_values.update(p._load_attr_values(attr_dict))
        return attr_values

    def __repr__(self):
        return 'MergeConfigParser(%r)' % self.parsers


class PyFileConfigParser(ConfigParser):
    """Config parser that loads values from Python files.

    This class will execute external Python files.  Then, all values with
    upper-cased names, which are produced by those Python files, will be
    matched against the attribute dict.  If any of the value names matches
    any of the attribute names, regardless of letter-case, then these values
    will be assigned to corresponding attributes.

    Parameters
    ----------
    files : str | collections.Iterable[str]
        List of Python files to be executed.  Values should be overwritten
        in the order specified by this list.

    ignore_not_exist : bool
        Whether or not to ignore a file if it does not exist?
        If not, the parser will raise ``IOError`` if a file does not exist.

    warn_unexpected : bool
        Whether or not to show warnings if unexpected values are specified?
    """

    def __init__(self, files, ignore_not_exist=True, warn_unexpected=False):
        if isinstance(files, str):
            self.python_files = [files]
        else:
            self.python_files = list(files)
        self.ignore_not_exist = ignore_not_exist
        self.warn_unexpected = warn_unexpected

    def _execute_file(self, path, locals_dict):
        with open(path) as f:
            code = compile(f.read(), path, 'exec')
            exec(code, {}, locals_dict)

    def _load_attr_values(self, attr_dict):
        # first, execute all files
        locals_dict = {}
        for path in self.python_files:
            if os.path.exists(path):
                self._execute_file(path, locals_dict)
            elif not self.ignore_not_exist:
                raise IOError(
                    'Config file %r is expected, but does not exist.' %
                    path
                )

        # next, try to pick out values that are needed.
        attr_values = {}
        for name, value in six.iteritems(locals_dict):
            if name.isupper():
                attr_name = name.lower()
                if attr_name in attr_dict:
                    attr_values[attr_name] = value
                elif self.warn_unexpected:
                    warnings.warn('Unexpected attribute %r.' % attr_name)

        return attr_values

    def __repr__(self):
        return 'PyFileConfigParser(%r)' % self.python_files


class EnvVarConfigParser(ConfigParser):
    """Config parser that loads values from environment variables.

    Parameters
    ----------
    prefix : str
        Common prefix for the config environment variables.
        All other variables without the prefix will be ignored.

    exclude_vars : collections.Iterable[str]
        List of environment variables to be excluded.

    warn_unexpected : bool
        Whether or not to show warnings if unexpected values are specified?
    """

    def __init__(self, prefix, exclude_vars=None, warn_unexpected=False):
        self.prefix = prefix
        self.exclude_vars = set(s.upper() for s in exclude_vars or ())
        self.warn_unexpected = warn_unexpected

    def _load_attr_values(self, attr_dict):
        attr_values = {}
        for attr_name, attr in six.iteritems(attr_dict):
            env_name = '%s%s' % (self.prefix, attr.name.upper())
            if env_name in os.environ and env_name not in self.exclude_vars:
                attr_values[attr_name] = os.environ[env_name]
        return attr_values

    def __repr__(self):
        return 'EnvVarConfigParser(%r)' % self.prefix


class DefaultConfigParser(MergeConfigParser):
    """Config parser that loads values from typical external sources.

    This parser will load values from the following sources:

    * Config file under user directory, located at:
        ~/.[app-name]rc     (Linux)
        ~/[app-name].conf   (Windows)
    * The first encountered config file ``[app-name].conf``, searched in all
      parent directories of the main script.
    * Config file specified by environment variable ``[APP_NAME]_CONFIG_FILE``.
    * Environment variables, with prefix ``[APP_NAME]_``.

    Parameters
    ----------
    app_name : str
        Application name, used to determine the location of config files,
        and the prefix of environment variables.

    warn_unexpected : bool
        Whether or not to show warnings if unexpected values are specified?
    """

    @classmethod
    def get_user_config_path(cls, app_name):
        """Get the path of user config file for specified application."""
        app_name = app_name.lower()
        if sys.platform.startswith('win'):
            return os.path.expanduser('~/%s.conf' % app_name)
        else:
            return os.path.expanduser('~/.%src' % app_name)

    def __init__(self, app_name, warn_unexpected=False):
        lower_name = app_name.lower()
        upper_name = app_name.upper()
        config_files = []
        parsers = []

        # include user directory config and main script config
        config_files.append(self.get_user_config_path(app_name))

        main_module = sys.modules['__main__']
        if hasattr(main_module, '__file__'):
            main_script = os.path.abspath(main_module.__file__)
            for parent in misc.yield_parent_dirs(main_script):
                config_file = os.path.join(parent, '%s.conf' % lower_name)
                if os.path.isfile(config_file):
                    config_files.append(config_file)
                    break

        parsers.append(
            PyFileConfigParser(
                config_files,
                ignore_not_exist=True,
                warn_unexpected=warn_unexpected
            )
        )

        # include the config file specified by environment variable
        app_config_file_env = '%s_CONFIG_FILE' % upper_name
        if os.environ.get(app_config_file_env):
            parsers.append(
                PyFileConfigParser(
                    [os.environ[app_config_file_env]],
                    ignore_not_exist=True,
                    warn_unexpected=warn_unexpected
                )
            )

        # include the environment variable parser
        app_env_prefix = '%s_' % upper_name
        parsers.append(
            EnvVarConfigParser(
                app_env_prefix,
                exclude_vars=[app_config_file_env],
                warn_unexpected=warn_unexpected
            )
        )

        # finally, initialize the merge parser
        super(DefaultConfigParser, self).__init__(parsers)
