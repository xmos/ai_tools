# Copyright (c) 2020, XMOS Ltd, All rights reserved

from typing import TYPE_CHECKING, Optional

from . import Buffer, _BufferOwnerContainer

if TYPE_CHECKING:
    from . import XCOREModel


class Metadata(_BufferOwnerContainer):
    buffer: Buffer["Metadata"]

    def __init__(
        self,
        model: "XCOREModel",
        name: str,
        buffer: Optional[Buffer["Metadata"]] = None,
    ) -> None:
        # Generally, do not use this constructor to instantiate Metadata!
        # Use XCOREModel.create_metadata instead.

        super().__init__(name)
        self._model = model  # parent

    @property
    def model(self) -> "XCOREModel":
        return self._model

    def __str__(self) -> str:
        return f"name={self.name}, buffer={self.buffer}"

    def sanity_check(self) -> None:
        assert self in self.buffer.owners
