from xmos_ai_tools import xformer
from xmos_ai_tools.xinterpreters import TFLMHostInterpreter
from ultralytics import YOLO
import numpy as np

HEIGHT, WIDTH = 160, 160
TFLITE_MODEL_PATH = "yolov8n-cls_saved_model/yolov8n-cls_full_integer_quant.tflite"
OPT_MODEL_PATH = "src/model.tflite"
OPT_PARAMS_PATH = "src/model_flash.params"
NAMING_PREFIX = "model_"

###############################################
# Creating and converting an YoloV8 cls model #
###############################################

# Load a model
model = YOLO("yolov8n-cls.pt")  # load an official model

# Export the model
_format = "tflite"

model.export(format=_format, imgsz=(HEIGHT, WIDTH), int8=True)

# Convert the model to XCore optimized TFLite via xformer:
# There are various ways to configure the compiler to optimize the model,
# operator splitting isn't documented yet. This configuration works well for
# MobileNetV2, reach out if you need assistance with other complex models
xformer.convert(
    TFLITE_MODEL_PATH,
    OPT_MODEL_PATH,
    {
        "xcore-flash-image-file": OPT_PARAMS_PATH,
        "xcore-thread-count": "5",
        "xcore-naming-prefix": NAMING_PREFIX,
    },
)

# Generate flash binary
xformer.generate_flash(
    output_file="xcore_flash_binary.out",
    model_files=[OPT_MODEL_PATH],
    param_files=[OPT_PARAMS_PATH],
)

#######################################################################
# Running the model on xcore host interpreter with sample input image #
#######################################################################

# Sample image of a lion (ImageNet class 291)
with open("lion.bin", "rb") as f:
    data = f.read()

input_array = np.frombuffer(data, dtype=np.uint8)
# input image values are in the range 0 to 255
# we subtract 128 to change to -128 to 127 for int8
input_array = (input_array - 128).astype(np.int8)


interpreter = TFLMHostInterpreter()
interpreter.set_model(model_path=OPT_MODEL_PATH, params_path=OPT_PARAMS_PATH)
interpreter.allocate_tensors()

# Interpreter.get_input_details and interpreter.get_output_details
# return a list for each input/output in the model
# MobileNetV2 only has a single input and output, so we unwrap it
(input_details,) = interpreter.get_input_details()
(output_details,) = interpreter.get_output_details()

input_data = input_array.astype(input_details["dtype"])
input_data = np.reshape(input_data, input_details["shape"])
interpreter.set_tensor(input_details["index"], input_data)

# Inference
interpreter.invoke()
detections = interpreter.get_tensor(output_details["index"])
print(f"Inferred imagenet class = {detections.argmax()}")
