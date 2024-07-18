from xmos_ai_tools import xformer

TFLITE_MODEL_PATH = "mobilenetv1_25.tflite"
OPTIMIZED_MODEL_PATH = "src/model.tflite"

OPTIMIZED_MODEL_PATH = "src/model.tflite"
WEIGHT_PARAMS_PATH = "src/model_weights.h"
print("Generating app cpp files for model...")
xformer.convert(
    TFLITE_MODEL_PATH,
    OPTIMIZED_MODEL_PATH,
    {
        "xcore-thread-count": "5",
        # set conv err threshold
        "xcore-conv-err-threshold": "0.6",
        # operation splitting to reduce tensor arena size
        "xcore-op-split-tensor-arena": "True",
        "xcore-op-split-top-op": "0",
        "xcore-op-split-bottom-op": "4",
        "xcore-op-split-num-splits": "10",
        # write weights as a header file to be 
        # placed on second tile
        "xcore-load-tile" : "True",
        # roughly(tensors are not split) specifies 
        # size of weights to move out to header file
        "xcore-max-load-external-size" : "270000",
        # move weights to this file
        "xcore-weights-file" : WEIGHT_PARAMS_PATH,
    },
)
xformer.print_optimization_report()

print("Done!")
