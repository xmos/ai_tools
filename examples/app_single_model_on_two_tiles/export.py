from pathlib import Path

from xmos_ai_tools import xformer as xf

cwd = Path(__file__).parent
model_in = cwd / "mobilenetv1_25.tflite"
model_out = cwd / "src/model.tflite"
params_file = cwd / "src/model_weights"
params = [
        ("xcore-thread-count", "5"),
        # set conv err threshold
        ("xcore-conv-err-threshold", "0.6"),
        # operation splitting to reduce tensor arena size
        ("xcore-op-split-tensor-arena", "True"),
        ("xcore-op-split-top-op", "0"),
        ("xcore-op-split-bottom-op", "4"),
        ("xcore-op-split-num-splits", "10"),
        # write weights as a C array to be served from the other tile
        ("xcore-write-weights-as-array", "True"),
        # roughly (tensors are not split) specifies
        # size of weights to move out to the generated array
        ("xcore-load-externally-if-larger", "1500"),
        ("xcore-max-load-external-size", "270000"),
        # move weights to files with this base name
        ("xcore-weights-file", params_file),
]
print("Generating app cpp files for model...")
xf.convert(model_in, model_out, params)
xf.print_optimization_report()

print("Done!")
