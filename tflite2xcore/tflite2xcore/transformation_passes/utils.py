# Copyright (c) 2019, XMOS Ltd, All rights reserved

import numpy as np

from functools import wraps

from tflite2xcore import utils


class Log(utils.Log):
    @classmethod
    def output(cls, func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            out = func(self, *args, **kwargs)
            msg = f"{func.__name__} output:\n"
            if isinstance(out, np.ndarray):
                msg += cls._array_msg(out, func.__name__)
            else:
                msg += f"{out}\n"
            self.logger.debug(msg)
            return out
        return wrapper
