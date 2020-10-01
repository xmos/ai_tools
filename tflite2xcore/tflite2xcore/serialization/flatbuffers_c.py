# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import struct
import ctypes
import numpy as np  # type: ignore
from enum import Enum

from tflite2xcore import libtflite2xcore as lib


class FlexbufferBuilder:
    def __init__(self, data=None):
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

    def _add_vector(self, obj, data, key=None):
        size = lib.builder_start_vector(obj, key)
        for list_item in data:
            if isinstance(list_item, Enum):
                list_item = list_item.value

            list_item_type = type(list_item)
            if list_item_type in (int, np.int32):
                lib.builder_vector_int(obj, int(list_item))
            elif list_item_type in (bool, np.bool):
                lib.builder_vector_bool(obj, bool(list_item))
            elif list_item_type in (float, np.float32):
                lib.builder_vector_float(obj, float(list_item))
            elif list_item_type is str:
                lib.builder_vector_string(obj, list_item.encode("ascii"))
            elif list_item_type is dict:
                self._add_map(obj, list_item)
            elif list_item_type in (list, tuple):
                self._add_vector(obj, list(list_item))
            else:
                raise Exception(
                    f"Type {list_item_type} not supported (list item={list_item})"
                )
        size = lib.builder_end_vector(self.obj, size, False, False)

        return size

    def _add_map(self, obj, data, key=None):
        msize = lib.builder_start_map(obj, key)

        for key, value in data.items():
            key_ascii = key.encode("ascii")
            if isinstance(value, Enum):
                value = value.value

            value_type = type(value)
            if value_type is int:
                lib.builder_int(obj, key_ascii, value)
            elif value_type is bool:
                lib.builder_bool(obj, key_ascii, value)
            elif value_type is float:
                lib.builder_float(obj, key_ascii, value)
            elif value_type is str:
                lib.builder_string(obj, key_ascii, value.encode("ascii"))
            elif value_type is dict:
                self._add_map(obj, value, key_ascii)
            elif value_type in (list, tuple):
                self._add_vector(obj, list(value), key_ascii)
            else:
                raise Exception(
                    f"Type {value_type} not supported (key={key_ascii}, value={value})"
                )

        size = lib.builder_end_map(obj, msize)
        return size

    def set_data(self, data):
        lib.builder_clear(self.obj)

        self._add_map(self.obj, data)

        lib.builder_finish(self.obj)

    def get_bytes(self, size=1024):
        buf = ctypes.create_string_buffer(size)
        actual_size = lib.builder_get_buffer(self.obj, buf)
        return [ubyte[0] for ubyte in struct.iter_unpack("B", buf[0:actual_size])]


class FlexbufferParser:
    def __init__(self):
        lib.parse_flexbuffer.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        lib.parse_flexbuffer.restype = ctypes.c_size_t

    def parse(self, buffer, size=100000):
        char_array = ctypes.c_char * len(buffer)
        json_buffer = ctypes.create_string_buffer(size)

        actual_size = lib.parse_flexbuffer(
            char_array.from_buffer_copy(buffer), len(buffer), json_buffer, size
        )

        return json_buffer[0:actual_size]
