# Copyright 2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import json
import struct
import ctypes
import numpy as np
from enum import Enum
from typing import Dict, Any, List, Optional, Union

from tflite2xcore import libflexbuffers as lib


class FlexbufferBuilder:
    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        lib.new_builder.restype = ctypes.c_void_p

        lib.builder_clear.argtypes = [ctypes.c_void_p]
        lib.builder_clear.restype = ctypes.c_void_p

        lib.builder_start_map.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.builder_start_map.restype = ctypes.c_size_t

        lib.builder_end_map.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.builder_string.restype = ctypes.c_size_t

        lib.builder_start_vector.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.builder_string.restype = ctypes.c_size_t

        lib.builder_end_vector.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_bool,
            ctypes.c_bool,
        ]
        lib.builder_string.restype = ctypes.c_size_t

        lib.builder_int.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        lib.builder_int.restype = ctypes.c_void_p

        lib.builder_uint.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
        lib.builder_uint.restype = ctypes.c_void_p

        lib.builder_bool.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool]
        lib.builder_bool.restype = ctypes.c_void_p

        lib.builder_float.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_float]
        lib.builder_float.restype = ctypes.c_void_p

        lib.builder_string.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        lib.builder_string.restype = ctypes.c_void_p

        lib.builder_vector_int.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.builder_vector_int.restype = ctypes.c_void_p

        lib.builder_vector_bool.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        lib.builder_vector_bool.restype = ctypes.c_void_p

        lib.builder_vector_float.argtypes = [ctypes.c_void_p, ctypes.c_float]
        lib.builder_vector_float.restype = ctypes.c_void_p

        lib.builder_vector_string.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.builder_vector_string.restype = ctypes.c_void_p

        lib.builder_finish.argtypes = [ctypes.c_void_p]
        lib.builder_finish.restype = ctypes.c_void_p

        lib.builder_get_buffer.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.builder_get_buffer.restype = ctypes.c_size_t

        self.obj = lib.new_builder()

        if data:
            self.set_data(data)

    def _add_vector(self, data: List[Any], key_ascii: Optional[bytes] = None) -> int:
        size = lib.builder_start_vector(self.obj, key_ascii)
        for list_item in data:
            if isinstance(list_item, Enum):
                list_item = list_item.value

            list_item_type = type(list_item)  # TODO: fix this
            if np.issubdtype(list_item_type, np.signedinteger):
                lib.builder_vector_int(self.obj, int(list_item))
            elif np.issubdtype(list_item_type, np.bool_):
                lib.builder_vector_bool(self.obj, bool(list_item))
            elif np.issubdtype(list_item_type, np.floating):
                lib.builder_vector_float(self.obj, float(list_item))
            elif list_item_type is str:
                lib.builder_vector_string(self.obj, list_item.encode("ascii"))
            elif list_item_type is dict:
                self._add_map(list_item)
            elif list_item_type in (list, tuple, np.ndarray):
                self._add_vector(list(list_item))
            else:
                raise Exception(
                    f"Type {list_item_type} not supported (list item={list_item})"
                )
        size = lib.builder_end_vector(self.obj, size, False, False)

        return size  # type: ignore

    def _add_map(self, data: Dict[str, Any], key_ascii: Optional[bytes] = None) -> int:
        msize = lib.builder_start_map(self.obj, key_ascii)

        for key, value in data.items():
            key_ascii = key.encode("ascii")
            if isinstance(value, Enum):
                value = value.value

            value_type = type(value)
            if np.issubdtype(value_type, np.signedinteger):
                lib.builder_int(self.obj, key_ascii, int(value))
            elif np.issubdtype(value_type, np.bool_):
                lib.builder_bool(self.obj, key_ascii, bool(value))
            elif np.issubdtype(value_type, np.floating):
                lib.builder_float(self.obj, key_ascii, float(value))
            elif value_type is str:
                lib.builder_string(self.obj, key_ascii, value.encode("ascii"))
            elif value_type is dict:
                self._add_map(value, key_ascii)
            elif value_type in (list, tuple, np.ndarray):
                self._add_vector(list(value), key_ascii)
            else:
                raise Exception(
                    f"Type {value_type} not supported (key={key}, value={value})"
                )

        size = lib.builder_end_map(self.obj, msize)
        return size  # type: ignore

    def set_data(self, data: Dict[str, Any]) -> None:
        lib.builder_clear(self.obj)

        self._add_map(data)

        lib.builder_finish(self.obj)

    def get_bytes(self, size: int = 1024) -> List[bytes]:
        buf = ctypes.create_string_buffer(size)
        actual_size = lib.builder_get_buffer(self.obj, buf)
        return [ubyte[0] for ubyte in struct.iter_unpack("B", buf[0:actual_size])]  # type: ignore


class FlexbufferParser:
    def __init__(self) -> None:
        lib.parse_flexbuffer.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        lib.parse_flexbuffer.restype = ctypes.c_size_t

    def parse(self, buffer: bytes, size: int = 100000) -> Any:
        if not buffer:
            return {}

        char_array = ctypes.c_char * len(buffer)
        json_buffer = ctypes.create_string_buffer(size)

        actual_size = lib.parse_flexbuffer(
            char_array.from_buffer_copy(buffer), len(buffer), json_buffer, size
        )

        return json.loads(json_buffer[0:actual_size])  # type: ignore
