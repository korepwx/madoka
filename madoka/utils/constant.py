# -*- coding: utf-8 -*-

__all__ = ['DEFAULT_TRAIN_BATCH_SIZE', 'DEFAULT_PREDICT_BATCH_SIZE',
           'INT32_MAX_VALUE', 'TrainingPhase', 'NamedStatusCode']

DEFAULT_TRAIN_BATCH_SIZE = 64
DEFAULT_PREDICT_BATCH_SIZE = 512
INT32_MAX_VALUE = (1 << 31) - 1


class NamedStatusCode(object):
    """Base class for named status codes."""

    @classmethod
    def iter_statuses(cls):
        """Iterate through all the status names and codes.

        Yields
        -------
        (str, int)
            Status name and its corresponding code.
        """
        for k in dir(cls):
            if k.isupper():
                v = getattr(cls, k)
                yield (k, v)

    @classmethod
    def get_name(cls, code):
        """Get the name of specified code.

        Returns
        -------
        str | None
            Name of the status code, or None if the code is not defined.
        """
        for k, v in cls.iter_statuses():
            if v == code:
                return k
        return None


class TrainingPhase(NamedStatusCode):
    """Enumeration of training phase."""

    #: Phase is not set.
    NOT_SET = 0x0

    #: Training phase.
    TRAINING = 0x1

    #: Validation phase.
    VALIDATION = 0x2

    #: Testing phase.
    TESTING = 0x4
