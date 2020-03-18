# Copyright (c) 2019, XMOS Ltd, All rights reserved

import numpy as np

from functools import wraps


class Log():
    @classmethod
    def _log_array_msg(cls, arr, style=''):
        msg = f"numpy.ndarray (shape={arr.shape}, dtype={arr.dtype}):\n"
        if style.endswith('_shift_scale_arr'):
            msg += f"shift_pre:\n{arr[:, 0]}\n"
            msg += f"scale:\n{arr[:, 1]}\n"
            msg += f"shift_post:\n{arr[:, 2]}"
        else:
            msg += f"{arr}"
        return msg + '\n'

    @classmethod
    def output(cls, func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            out = func(self, *args, **kwargs)
            msg = f"{func.__name__} output:\n"
            if isinstance(out, np.ndarray):
                msg += cls._log_array_msg(out, func.__name__)
            else:
                msg += f"{out}\n"
            self.logger.debug(msg)
            return out
        return wrapper
