# Copyright (c) 2019, XMOS Ltd, All rights reserved

import numpy as np

from functools import wraps

from tflite2xcore import logging


def log_method_output(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        out = func(self, *args, **kwargs)
        msg = f"{func.__name__} output:\n"
        if isinstance(out, np.ndarray):
            msg += logging._array_msg(out, func.__name__)
        else:
            msg += f"{out}\n"
        self.logger.xdebug(msg)
        return out
    return wrapper
