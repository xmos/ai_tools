# Copyright (c) 2018-2020, XMOS Ltd, All rights reserved1
import sys
import ctypes
from pathlib import Path

__PARENT_DIR = Path(__file__).parent.absolute()
if sys.platform.startswith("linux"):
    lib_path = str(__PARENT_DIR / "libs" / "linux" / "libtflite2xcore.so")
elif sys.platform == "darwin":
    lib_path = str(__PARENT_DIR / "libs" / "macos" / "libtflite2xcore.dylib")
else:
    raise RuntimeError("tflite2xcore is not yet supported on Windows!")

libtflite2xcore = ctypes.cdll.LoadLibrary(lib_path)

from . import xcore_schema
from . import xcore_model
from . import interpreters
from . import converter
from . import pass_manager
from . import parallelization
from . import tflite_visualize
from . import utils
from . import analyze
from . import model_generation
