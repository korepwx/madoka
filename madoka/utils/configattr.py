# -*- coding: utf-8 -*-
import six

__all__ = [
    'ConfigAttr', 'TypedConfigAttr', 'StrConfigAttr', 'IntConfigAttr',
    'FloatConfigAttr', 'BoolConfigAttr',
]


class ConfigAttr(object):
    """Config attribute.

    The ``ConfigAttr`` instances are designed to be declared as class
    attributes.  These attribute instances can be accessed only from
    their owner classes.  Any attempt to read the class objects will
    result in attribute values.

    Parameters
    ----------
    name : str
        Name of this config attribute.
        Attribute names will be case-insensitive.

    default : any
        Default value of this config attribute.

    filter : (any) -> any
        Function to check the value assigned to this config.

    description : str
        Description of this config attribute.
    """

    def __init__(self, name, default, filter=None, description=None):
        self.name = name
        self.default = default
        self.filter = filter
        self.description = description

    def __get__(self, instance, owner):
        if instance is None:
            # in order for ``ConfigParser`` to be able to get the
            # attribute object, we should return the attribute
            # object itself if accessed from the class
            return self
        if not hasattr(self, 'val'):
            if callable(self.default):
                return self.default()
            else:
                return self.default
        return self.val

    def __set__(self, instance, value):
        if self.filter:
            self.val = self.filter(value)
        else:
            self.val = value


class TypedConfigAttr(ConfigAttr):
    """Config attribute with a pre-defined type T.

    Parameters
    ----------
    name : str
        Name of this config attribute.

    default : any
        Default value for this config attribute.

    converter : (any) -> T
        Function to convert from any value to instance of the T type.

    validator : (T) -> bool
        Function to check whether or not the specified value can be accepted.
        The validator will only be called after the converter is applied.

    description : str
        Description of this config attribute.
    """

    def __init__(self, name, default, converter, validator=None,
                 description=None):
        self.converter = converter

        def filter(val):
            casted = converter(val)
            if validator is not None:
                if not validator(casted):
                    raise ValueError(
                        'Invalid value (%r) for configuration attribute %r.' %
                        (val, self.name)
                    )
            return casted

        super(TypedConfigAttr, self).__init__(
            name, default, filter, description)


class StrConfigAttr(TypedConfigAttr):
    def __init__(self, name, default, validator=None, description=None):
        super(StrConfigAttr, self).__init__(
            name, default, str, validator, description)


class IntConfigAttr(TypedConfigAttr):
    def __init__(self, name, default, validator=None, description=None):
        super(IntConfigAttr, self).__init__(
            name, default, int, validator, description)


class FloatConfigAttr(TypedConfigAttr):
    def __init__(self, name, default, validator=None, description=None):
        super(FloatConfigAttr, self).__init__(
            name, default, float, validator, description)


class BoolConfigAttr(TypedConfigAttr):
    def __init__(self, name, default, validator=None, description=None):
        def tobool(val):
            if isinstance(val, six.string_types):
                val = val.lower()
            if val in ('false', '0', 0, False):
                return False
            if val in ('true', '1', 1, True):
                return True

        def valid(val):
            if val in (True, False):
                # note that the validator will only be called after the
                # converter has been applied, so we just need to match
                # `True` and `False`.
                if validator is not None:
                    return validator(val)
                return True
            return False

        super(BoolConfigAttr, self).__init__(
            name, default, tobool, valid, description)
