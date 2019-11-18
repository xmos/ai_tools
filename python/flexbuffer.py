#!/usr/bin/env python
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import sys
import os
import ctypes

if sys.platform.startswith("linux"):
    lib = ctypes.cdll.LoadLibrary('../flexbuffers_python/build/libtflite2xcore.so')
elif sys.platform == "darwin":
    lib = ctypes.cdll.LoadLibrary('../flexbuffers_python/build/libtflite2xcore.dylib')
else:
    lib = ctypes.cdll.LoadLibrary('../flexbuffers_python/build/tflite2xcore.dll')

class FlexbufferBuilder(object):
    def __init__(self, data=None):
        #lib.new_builder.argtypes = []
        lib.new_builder.restype = ctypes.c_void_p

        lib.builder_clear.argtypes = [ctypes.c_void_p]
        lib.builder_clear.restype = ctypes.c_void_p

        lib.builder_start_map.argtypes = [ctypes.c_void_p]
        lib.builder_string.restype = ctypes.c_size_t

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
            self.Set(data)

    def Set(self, data):
        lib.builder_clear(self.obj)

        msize = lib.builder_start_map(self.obj)

        for key, value in data.items():
            key_ascii = key.encode('ascii')
            value_type = type(value)
            if value_type == int:
                lib.builder_int(self.obj, key_ascii, value)
            elif value_type == str:
                lib.builder_string(self.obj, key_ascii, value.encode('ascii'))
            elif value_type == list:
                vsize = lib.builder_start_vector(self.obj, key_ascii)
                for list_item in value:
                    lib.builder_vector_int(self.obj, list_item)
                vsize = lib.builder_end_vector(self.obj, vsize, False, False)
            else:
                raise Exception(f'Type {value_type} not supported')


        size = lib.builder_end_map(self.obj, msize)

        lib.builder_finish(self.obj)
    
    def GetBuffer(self, size=1024):
        buf = ctypes.create_string_buffer(size)
        actual_size = lib.builder_get_buffer(self.obj, buf)
        return buf[0:actual_size]

#*********************
# Example usage
#*********************
bits = {
    'foo': 12345,
    'bar': [1, 2, 3, 4, 5],
    'fizz': 'buzz'
}

print('***************')
print('* Data ')
print('***************')
print(bits)

print('***************')
print('*  Flexbuffer')
print('***************')
fbb = FlexbufferBuilder(bits)
buf = fbb.GetBuffer()
print(buf)
