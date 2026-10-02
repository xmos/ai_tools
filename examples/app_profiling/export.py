# Copyright 2026 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from pathlib import Path

from xmos_ai_tools import xformer as xf


cwd = Path(__file__).parent
model_in = cwd / "vww_quant.tflite"
model_out = cwd / "src/model.tflite"
params = []

assert model_in.exists(), f"Model not found: {model_in}"
xf.convert(model_in, model_out, params)