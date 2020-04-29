# Copyright (c) 2018-2020, XMOS Ltd, All rights reserved

from functools import wraps

import logging
from logging import NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL


XDEBUG = DEBUG - 1
VERBOSITIES = [WARNING, INFO, DEBUG, XDEBUG]

logging.addLevelName(XDEBUG, "XDEBUG")


def set_verbosity(verbosity=0):
    assert verbosity >= 0
    verbosity = min(verbosity, len(VERBOSITIES) - 1)

    logging.basicConfig(level=VERBOSITIES[verbosity])
    if verbosity == 0:
        logging.getLogger("tensorflow").setLevel(logging.ERROR)


def _array_msg(arr, style=""):
    msg = f"numpy.ndarray, shape={arr.shape}, dtype={arr.dtype}:\n"
    if style.endswith("_shift_scale_arr"):
        msg += f"shift_pre:\n{arr[:, 0]}\n"
        msg += f"scale:\n{arr[:, 1]}\n"
        msg += f"shift_post:\n{arr[:, 2]}"
    else:
        msg += f"{arr}"
    return msg + "\n"


def getLogger(*args, **kwargs):
    logger = logging.getLogger(*args, *kwargs)
    logger.xdebug = lambda *args, **kwargs: logger.log(XDEBUG, *args, **kwargs)
    return logger


class LoggingContext:
    def __init__(self, logger, level=None, handler=None, close=True):
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, exception_type, exception_value, traceback):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()


def log_method_output(level=XDEBUG, logger=None):
    def _log_method_output(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            out = func(self, *args, **kwargs)
            msg = f"{func.__name__} output:\n"
            import numpy as np

            if isinstance(out, np.ndarray):
                msg += _array_msg(out, func.__name__)
            else:
                msg += f"{out}\n"
            _logger = logger or self.logger
            _logger.log(level, msg)
            return out

        return wrapper

    return _log_method_output
