from xmos_ai_tools import xformer
from xmos_ai_tools.xinterpreters import TFLMHostInterpreter
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from save_mobilenet import save_quantized_mobilenet
import numpy as np

HEIGHT, WIDTH, CHANNELS = 160, 160, 3
TFLITE_MODEL_PATH = "mobilenetv2.tflite"
OPT_MODEL_PATH = "src/model.tflite"
OPT_PARAMS_PATH = "src/model_flash.params"
NAMING_PREFIX = "model_"
ALPHA_VALUE = 1.0

###############################################
# Creating and converting a MobileNetV2 model #
###############################################

# Obtain MobileNetV2 model
def get_mobilenetv2():
    model = MobileNetV2(
        input_shape=(HEIGHT, WIDTH, CHANNELS),
        alpha=ALPHA_VALUE,
        weights="imagenet",
    )
    save_quantized_mobilenet(model, TFLITE_MODEL_PATH, (HEIGHT, WIDTH))

# Quantize the model to int8 and save

# Convert the model to XCore optimized TFLite via xformer:
# There are various ways to configure the compiler to optimize the model,
# operator splitting isn't documented yet. This configuration works well for
# MobileNetV2, reach out if you need assistance with other complex models

def optimize_mobilenetv2():
    xformer.convert(
        TFLITE_MODEL_PATH,
        OPT_MODEL_PATH,
        [
            ("xcore-weights-file", OPT_PARAMS_PATH),
            ("xcore-thread-count", "5"),
            ("xcore-naming-prefix", NAMING_PREFIX),
            ("xcore-op-split-tensor-arena", "True"),
            ("xcore-op-split-top-op", "0"),
            ("xcore-op-split-bottom-op", "10"),
            ("xcore-op-split-num-splits", "7"),
            ("xcore-conv-err-threshold", "3"),
        ],
    )

    # Generate flash binary
    xformer.generate_flash(
        output_file="xcore_flash_binary.out",
        model_files=[OPT_MODEL_PATH],
        param_files=[OPT_PARAMS_PATH],
    )

    return xformer.tensor_arena_size()


if __name__ == "__main__":
    #######################################################################
    # Running the model on xcore host interpreter with sample input image #
    #######################################################################
    get_mobilenetv2()
    optimize_mobilenetv2()

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
