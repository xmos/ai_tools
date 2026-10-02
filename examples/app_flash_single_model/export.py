# Copyright 2026 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from pathlib import Path

from xmos_ai_tools import xformer as xf


cwd = Path(__file__).parent
model_in = cwd / "vww_quant.tflite"
model_out = cwd / "src/model.tflite"
params_file = cwd / "model.params"
flash_out = cwd / "xcore_flash_binary.out"
params = [("xcore-weights-file", params_file)]

assert model_in.exists(), f"Model not found: {model_in}"
xf.convert(model_in, model_out, params)

xf.generate_flash(
    output_file=flash_out,
    model_files=[model_out],
    param_files=[params_file],
)
