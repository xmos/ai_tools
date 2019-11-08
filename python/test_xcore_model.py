#!/usr/bin/env python
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved


import ctypes

lib = ctypes.cdll.LoadLibrary('/Users/keith/repos/hotdog/ai_tools/tflite2xcore/build/libtflite2xcore.dylib')


model = lib.model_import('/Users/keith/repos/hotdog/ai_tools/python/examples/arm_benchmark/models/model_xcore.tflite')