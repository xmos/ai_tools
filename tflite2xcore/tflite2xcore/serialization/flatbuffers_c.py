# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import sys
import os
import struct
from pathlib import Path
import ctypes

if sys.platform.startswith("linux"):
    shared_lib = os.path.join(Path(__file__).parent.absolute(), 'linux/libtflite2xcore.so.1.0.1')
elif sys.platform == "darwin":
    shared_lib = os.path.join(Path(__file__).parent.absolute(), 'macos/libtflite2xcore.1.0.1.dylib')
else:
    shared_lib = os.path.join(Path(__file__).parent.absolute(), 'windows/libtflite2xcore.dll')

lib = ctypes.cdll.LoadLibrary(shared_lib)

class FlatbufferIO:
    def __init__(self, data=None):
        lib.read_flatbuffer.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.read_flatbuffer.restype = ctypes.c_size_t

        lib.write_flatbuffer.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.write_flatbuffer.restype = ctypes.c_size_t

    def read_flatbuffer(self, schema, fbs, size=100000000):
        buffer = ctypes.create_string_buffer(size)
        actual_size = lib.read_flatbuffer(schema.encode('ascii'), fbs.encode('ascii'), buffer)

        return buffer[0:actual_size]

    def write_flatbuffer(self, schema, buffer, fbs):
        actual_size = lib.write_flatbuffer(schema.encode('ascii'), buffer, fbs.encode('ascii'))
        return actual_size

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

        lib.builder_end_vector.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_bool, ctypes.c_bool]
        lib.builder_string.restype = ctypes.c_size_t

        lib.builder_int.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        lib.builder_int.restype = ctypes.c_void_p

        lib.builder_uint.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
        lib.builder_uint.restype = ctypes.c_void_p

        lib.builder_string.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.builder_string.restype = ctypes.c_void_p

        lib.builder_vector_int.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.builder_vector_int.restype = ctypes.c_void_p

        lib.builder_finish.argtypes = [ctypes.c_void_p]
        lib.builder_finish.restype = ctypes.c_void_p

        lib.builder_get_buffer.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.builder_get_buffer.restype = ctypes.c_size_t

        self.obj = lib.new_builder()

        if data:
            self.set_data(data)

    def __add_vector(self, obj, data, key=None):
        size = lib.builder_start_vector(obj, key)
        for list_item in data:
            list_item_type = type(list_item)
            if list_item_type == int:
                lib.builder_vector_int(obj, list_item)
            elif list_item_type == dict:
                self.__add_map(obj, list_item)
            elif list_item_type == list:
                self.__add_vector(obj, list_item)
            else:
                raise Exception(f'Type {list_item_type} not supported')
        size = lib.builder_end_vector(self.obj, size, False, False)

        return size

    def __add_map(self, obj, data, key=None):
        msize = lib.builder_start_map(obj, key)

        for key, value in data.items():
            key_ascii = key.encode('ascii')
            value_type = type(value)
            if value_type == int:
                lib.builder_int(obj, key_ascii, value)
            elif value_type == str:
                lib.builder_string(obj, key_ascii, value.encode('ascii'))
            elif value_type == dict:
                self.__add_map(obj, value, key_ascii)
            elif value_type == list:
                self.__add_vector(obj, value, key_ascii)
            else:
                raise Exception(f'Type {value_type} not supported')

        size = lib.builder_end_map(obj, msize)
        return size

    def set_data(self, data):
        lib.builder_clear(self.obj)

        self.__add_map(self.obj, data)

        lib.builder_finish(self.obj)
    
    def get_bytes(self, size=1024):
        buf = ctypes.create_string_buffer(size)
        actual_size = lib.builder_get_buffer(self.obj, buf)
        return [ubyte[0] for ubyte in struct.iter_unpack('B', buf[0:actual_size])]

class FlexbufferParser:
    def __init__(self):
        lib.parse_flexbuffer.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        lib.parse_flexbuffer.restype = ctypes.c_size_t


    def parse(self, buffer, size=100000):
        char_array = ctypes.c_char * len(buffer)
        json_buffer = ctypes.create_string_buffer(size)

        actual_size = lib.parse_flexbuffer(char_array.from_buffer_copy(buffer), len(buffer), json_buffer, size)
    
        return json_buffer[0:actual_size]
