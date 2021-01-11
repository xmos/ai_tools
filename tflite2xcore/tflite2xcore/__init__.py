# Copyright (c) 2018-2020, XMOS Ltd, All rights reserved
import sys
import ctypes

from pathlib import Path

__PARENT_DIR = Path(__file__).parent.absolute()
if sys.platform.startswith("linux"):
    lib_path = str(__PARENT_DIR / "libs" / "linux" / "libflexbuffers.so")
elif sys.platform == "darwin":
    lib_path = str(__PARENT_DIR / "libs" / "macos" / "libflexbuffers.dylib")
else:
    raise RuntimeError("tflite2xcore is not yet supported on Windows!")

libflexbuffers = ctypes.cdll.LoadLibrary(lib_path)
from . import version

__version__ = version.get_version()

from . import xcore_schema
from . import xcore_model
from . import execution_planning
from . import interpreters
from . import converter
from . import pass_manager
from . import parallelization
from . import tflite_visualize
from . import utils
from . import analyze
from . import model_generation
