from xmos_ai_tools import xformer

TFLITE_MODEL_PATH = "mobilenetv2.tflite"
OPTIMIZED_MODEL_PATH = "src/model.tflite"

OPTIMIZED_MODEL_PATH = "src/model.tflite"
WEIGHT_PARAMS_PATH = "src/model_weights"
FLASH_IMAGE_PATH = "src/xcore_flash_binary.out"
print("Generating app cpp files for model...")
xformer.convert(
    TFLITE_MODEL_PATH,
    OPTIMIZED_MODEL_PATH,
    {
        "xcore-thread-count": "5",
        # set conv err threshold
        "xcore-conv-err-threshold": "3",
        # operation splitting to reduce tensor arena size
        "xcore-op-split-tensor-arena": "True",
        "xcore-op-split-top-op": "0,7",
        "xcore-op-split-bottom-op": "6,14",
        "xcore-op-split-num-splits": "8,4",
        # write weights as an array to be placed in DDR
        "xcore-write-weights-as-array" : "True",
        "xcore-weights-in-external-memory" : "True",
        # For DDR, we want to ideally reduce loads smaller
        # than 4000 bytes, as they are slower.
        # But this would increase memory usage on tile
        # and so it is a tradeoff
        "xcore-load-externally-if-larger" : "1500",
        # move weights to this file
        "xcore-weights-file" : WEIGHT_PARAMS_PATH,
    },
)
xformer.print_optimization_report()

# # Generate flash image to be flashed using xflash
# xformer.generate_flash(
#     output_file=FLASH_IMAGE_PATH,
#     model_files=[OPTIMIZED_MODEL_PATH],
#     param_files=[WEIGHT_PARAMS_PATH],
# )

print("Done!")
