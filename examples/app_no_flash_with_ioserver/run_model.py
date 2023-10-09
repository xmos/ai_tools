from xmos_ai_tools.xinterpreters import TFLMHostInterpreter
from xmos_ai_tools.io_server import IOServer
import numpy as np
import cv2


def img_to_arr(image_file, input_details):
    im = cv2.imread(image_file)
    h, w = im.shape[:2]

    # RGB conversion
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Resize
    im_rgb = cv2.resize(im, (input_details["shape"][1], input_details["shape"][2]))

    # uint8 to int8
    input_data = im_rgb.astype(np.float32)
    input_data -= 128.0
    input_data = input_data.astype(input_details["dtype"])
    input_data = np.expand_dims(input_data, axis=0)
    return input_data


def print_detections(detections):
    label = "Not human" if detections[0] > detections[1] else "Human"
    percentage = (detections[1] + 128) * 100 / 255
    print(f"{label} ({percentage:.0f}%)")


OPT_MODEL_PATH = "src/model.tflite"

########################################################################
# Running the model on xcore host interpreter with sample input images #
########################################################################

# Setup host interpreter
interpreter = TFLMHostInterpreter()
interpreter.set_model(model_path=OPT_MODEL_PATH)
interpreter.allocate_tensors()

# Interpreter.get_input_details and interpreter.get_output_details
# return a list for each input/output in the model
# MobileNetV2 only has a single input and output, so we unwrap it
(input_details,) = interpreter.get_input_details()
(output_details,) = interpreter.get_output_details()

# Read input image and convert to array
input_data = img_to_arr("human.jpg", input_details)
interpreter.set_tensor(input_details["index"], input_data)

# Inference
interpreter.invoke()
(detections,) = interpreter.get_tensor(output_details["index"])
print_detections(detections)

# Read input image and convert to array
input_data = img_to_arr("nonhuman.jpg", input_details)
interpreter.set_tensor(input_details["index"], input_data)

# Inference
interpreter.invoke()
(detections,) = interpreter.get_tensor(output_details["index"])
print_detections(detections)


########################################################################
# Running the model on xcore device with sample input images via USB #
########################################################################

# The app must be running on xcore so that it can be connected via USB
ie = IOServer()
ie.connect()

input_data = img_to_arr("human.jpg", input_details)
ie.write_input_tensor(input_data.tobytes())
ie.start_inference()
# ie.read_output_tensor returns an array with at least four bytes
# For this model, the output tensor is only two bytes
detections = ie.read_output_tensor(2)
print_detections(detections)

input_data = img_to_arr("nonhuman.jpg", input_details)
ie.write_input_tensor(input_data.tobytes())
ie.start_inference()
# ie.read_output_tensor returns a byte array with at least four bytes
# For this model, the output tensor is only two bytes
detections = ie.read_output_tensor(2)
print_detections(detections)
