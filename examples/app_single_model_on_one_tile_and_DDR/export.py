from pathlib import Path

from xmos_ai_tools import xformer as xf

cwd = Path(__file__).parent
model_in = cwd / "mobilenetv2.tflite"
model_out = cwd / "src/model.tflite"
params_file = cwd / "src/model_weights"
params = [
        ("xcore-thread-count", "5"),
        # set conv err threshold
        ("xcore-conv-err-threshold", "3"),
        # operation splitting to reduce tensor arena size
        ("xcore-op-split-tensor-arena", "True"),
        ("xcore-op-split-top-op", "0,7"),
        ("xcore-op-split-bottom-op", "6,14"),
        ("xcore-op-split-num-splits", "8,4"),
        # write weights as an array to be placed in DDR
        ("xcore-write-weights-as-array", "True"),
        ("xcore-weights-in-external-memory", "True"),
        # For DDR, we want to ideally reduce loads smaller
        # than 4000 bytes, as they are slower.
        # But this would increase memory usage on tile
        # and so it is a tradeoff
        ("xcore-load-externally-if-larger", "1500"),
        # move weights to this file
        ("xcore-weights-file", params_file),
]
print("Generating app cpp files for model...")
xf.convert(model_in, model_out, params)
xf.print_optimization_report()

print("Done!")
