# Copyright 2026 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from pathlib import Path

from xmos_ai_tools import xformer as xf


cwd = Path(__file__).parent
model_in = cwd / "vww_quant1.tflite"
model_out = cwd / "src/model1.tflite"
params_file = cwd / "model1.params"
params = [
    ("xcore-weights-file", params_file),
    ("xcore-naming-prefix", "model1_"),
]
assert model_in.exists(), f"Model not found: {model_in}"
xf.convert(model_in, model_out, params)

model_in = cwd / "vww_quant2.tflite"
model_out = cwd / "src/model2.tflite"
params_file = cwd / "model2.params"
params = [
    ("xcore-weights-file", params_file),
    ("xcore-naming-prefix", "model2_"),
]
assert model_in.exists(), f"Model not found: {model_in}"
xf.convert(model_in, model_out, params)

xf.generate_flash(
    output_file=cwd / "xcore_flash_binary.out",
    model_files=[cwd / "src/model1.tflite", cwd / "src/model2.tflite"],
    param_files=[cwd / "model1.params", cwd / "model2.params"],
)
