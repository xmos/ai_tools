# Copyright (c) 2020, XMOS Ltd, All rights reserved

from enum import IntEnum
from typing import Callable

from . import BuiltinOpCodes

class BuiltinOptions(IntEnum):
    # TODO: consider adding fields for IDE support
    from_numpy_dtype: Callable[[BuiltinOpCodes], BuiltinOptions]
