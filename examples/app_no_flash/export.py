from pathlib import Path

from xmos_ai_tools import xformer as xf


cwd = Path(__file__).parent
model_in = cwd / "vww_quant.tflite"
model_out = cwd / "src/model.tflite"

xf.convert(model_in, model_out, [])
