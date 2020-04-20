# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import sys
import os
import ctypes
from pathlib import Path

__PARENT_DIR = Path(__file__).parent.absolute()
if sys.platform.startswith("linux"):
    lib_path = os.path.join(__PARENT_DIR, "libs/linux/libtflite2xcore.so")
elif sys.platform == "darwin":
    lib_path = os.path.join(__PARENT_DIR, "libs/macos/libtflite2xcore.dylib")
else:
    raise RuntimeError("tflite2xcore is not yet supported on Windows!")

libtflite2xcore = ctypes.cdll.LoadLibrary(lib_path)

from . import converter
from . import pass_manager
from . import operator_codes
from . import parallelization
from . import tflite_visualize
from . import utils
from . import xlogging
from . import xcore_model
from . import analyze
