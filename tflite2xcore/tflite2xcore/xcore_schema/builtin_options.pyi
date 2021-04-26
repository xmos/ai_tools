# Copyright (c) 2020, XMOS Ltd, All rights reserved

from enum import IntEnum
from typing import Callable, Any

from . import BuiltinOpCodes

class BuiltinOptions(IntEnum):
    # TODO: consider adding fields for IDE support
    @classmethod
    def __call__(cls, x: Any) -> BuiltinOptions: ...
    from_BuiltinOpCodes: Callable[[BuiltinOpCodes], BuiltinOptions]
