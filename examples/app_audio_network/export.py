# Copyright 2026 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from pathlib import Path

from xmos_ai_tools import xformer as xf


cwd = Path(__file__).parent
model_in = cwd / "denoise_16x8.tflite"
model_out = cwd / "src/model_audioi16.tflite"
params = [("xcore-thread-count", 5)]

assert model_in.exists(), f"Model not found: {model_in}"
xf.convert(model_in, model_out, params)