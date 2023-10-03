from xmos_ai_tools import xformer

TFLITE_MODEL_PATH = "vww_quant.tflite"
OPT_MODEL_PATH = "src/model.tflite"
NAMING_PREFIX = "model_"

# Convert the model to XCore optimized TFLite via xformer:
xformer.convert(
    TFLITE_MODEL_PATH,
    OPT_MODEL_PATH,
    {
        "xcore-thread-count": "5",
        "xcore-naming-prefix": NAMING_PREFIX,
    },
)
