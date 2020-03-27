# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import sys
import os
import ctypes
from pathlib import Path

__PARENT_DIR = Path(__file__).parent.absolute()
if sys.platform.startswith("linux"):
    lib_path = os.path.join(__PARENT_DIR, 'libs/linux/libtflite2xcore.so.1.0.1')
elif sys.platform == "darwin":
    lib_path = os.path.join(__PARENT_DIR, 'libs/macos/libtflite2xcore.1.0.1.dylib')
else:
    lib_path = os.path.join(__PARENT_DIR, 'libs/windows/libtflite2xcore.dll')

libtflite2xcore = ctypes.cdll.LoadLibrary(lib_path)

from .serialization.api import (
    read_flatbuffer,
    write_flatbuffer,
    create_dict_from_model
)
