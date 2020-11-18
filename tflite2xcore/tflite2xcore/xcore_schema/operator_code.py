# Copyright (c) 2020, XMOS Ltd, All rights reserved

from typing import Optional, Union, Any

from . import XCOREOpCodes, ExternalOpCodes, BuiltinOpCodes

CustomOpCodes = Union[XCOREOpCodes, ExternalOpCodes]

ValidOpCodes = Union[BuiltinOpCodes, CustomOpCodes]


class OperatorCode:
    def __init__(self, opcode: ValidOpCodes, *, version: Optional[int] = None) -> None:
        self.version = version or 1
        self.code = opcode

    @property
    def name(self) -> str:
        return self.code.name

    @property
    def value(self) -> Union[int, str]:
        return self.code.value

    def __eq__(self, obj: Any) -> bool:
        return (
            isinstance(obj, OperatorCode)
            and obj.code is self.code
            and obj.version == self.version
        )

    def __hash__(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        return f"{self.code} (version {self.version})"
