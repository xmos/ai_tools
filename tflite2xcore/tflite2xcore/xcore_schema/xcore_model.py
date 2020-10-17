# Copyright (c) 2018-2020, XMOS Ltd, All rights reserved

from typing import (
    Any,
    Union,
    Optional,
    Iterable,
    List,
    Counter,
)

from . import (
    _IRObject,
    OperatorCode,
    Buffer,
    _BufferOwnerContainer,
    _BufferDataType,
    Subgraph,
    Metadata,
)
from .flatbuffers_io import XCORESerializationMixin


class XCOREModel(XCORESerializationMixin, _IRObject):
    def __init__(
        self,
        version: Optional[int] = None,
        description: Optional[str] = None,
        subgraphs: Optional[Iterable[Subgraph]] = None,
        buffers: Optional[Iterable[Buffer[_BufferOwnerContainer]]] = None,
        metadata: Optional[Iterable[Metadata]] = None,
    ) -> None:
        super().__init__()
        self.version = version or 3
        self.description = description or ""
        self.buffers = list(buffers or [])
        self.subgraphs = list(subgraphs or [])
        self.metadata = list(metadata or [])

    def is_equal(self, other: Any) -> bool:
        return (
            super().is_equal(other)
            and self.version == other.version
            # and self.description == other.description  # intentionally not compared
            and self.sequence_equal(self.buffers, other.buffers)
            and self.sequence_equal(self.subgraphs, other.subgraphs)
            and self.sequence_equal(self.metadata, other.metadata)
        )

    def create_buffer(
        self, data: Optional[_BufferDataType] = None
    ) -> Buffer[_BufferOwnerContainer]:
        buffer = Buffer[_BufferOwnerContainer](self, data)
        self.buffers.append(buffer)
        return buffer

    def create_metadata(
        self, name: str, buffer: Optional[Buffer[Metadata]] = None
    ) -> Metadata:
        metadata = Metadata(self, name)
        metadata.buffer = Metadata.create_buffer(self) if buffer is None else buffer
        metadata.buffer.owners.append(metadata)

        self.metadata.append(metadata)
        return metadata

    def create_subgraph(self, name: str = "") -> Subgraph:
        subgraph = Subgraph(self, name)
        self.subgraphs.append(subgraph)
        return subgraph

    def count_operator_codes(self) -> Counter[OperatorCode]:
        return Counter(
            operator.operator_code
            for subgraph in self.subgraphs
            for operator in subgraph.operators
        )

    @property
    def operator_codes(self) -> List[OperatorCode]:
        # sort the operators codes from most frequent to least frequent
        #   why? because the flatbuffer is a tiny bit smaller if we do
        return [op_code for op_code, _ in self.count_operator_codes().most_common()]

    @property
    def data_size(self) -> int:
        return sum(len(buffer) for buffer in self.buffers)

    def sanity_check(self) -> None:
        # check for duplicates
        assert len(self.subgraphs) == len(set(self.subgraphs))
        assert len(self.buffers) == len(set(self.buffers))
        assert len(self.metadata) == len(set(self.metadata))
        # the model is sane as long as all its subgraphs are sane
        for subgraph in self.subgraphs:
            subgraph.sanity_check()
        for buffer in self.buffers:
            buffer.sanity_check()
        for metadata in self.metadata:
            metadata.sanity_check()
