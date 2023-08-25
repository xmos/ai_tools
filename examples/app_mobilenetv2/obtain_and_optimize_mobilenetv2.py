from xmos_ai_tools import xformer
from xmos_ai_tools.xinterpreters import xcore_tflm_host_interpreter
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import numpy as np

HEIGHT, WIDTH = 160, 160
input_shape = (HEIGHT, WIDTH, 3)
alpha_value = 1.0

# Obtain MobileNet model
model = MobileNetV2(
    input_shape=input_shape,
    alpha=alpha_value,
    weights="imagenet",
)

# Use tf lite converter to quantize the model to int8
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Use a representative dataset generator for accurate activation quantization
rep_ds = tf.data.Dataset.list_files("image_samples/*.jpg")


def representative_dataset_gen():
    for image_path in rep_ds:
        img = tf.io.read_file(image_path)
        img = tf.io.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        resized_img = tf.image.resize(img, (HEIGHT, WIDTH))
        resized_img = resized_img[tf.newaxis, :]
        yield [resized_img]


converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()


model_path = "mobilenetv2.tflite"
with open(model_path, "wb") as f:
    f.write(tflite_model)
print("Model quantized and saved successfully.")


opt_model_path = "src/model.tflite"
opt_params_path = "src/model_flash.params"
naming_prefix = "model_"
xformer.convert(
    model_path,
    opt_model_path,
    {
        "xcore-flash-image-file": opt_params_path,
        "xcore-thread-count": "5",
        "xcore-naming-prefix": naming_prefix,
        "xcore-op-split-tensor-arena": "True",
        "xcore-op-split-top-op": "0",
        "xcore-op-split-bottom-op": "10",
        "xcore-op-split-num-splits": "7",
        "xcore-conv-err-threshold": "3",
    },
)


xformer.generate_flash(
    output_file="xcore_flash_binary.out",
    model_files=[opt_model_path],
    param_files=[opt_params_path],
)


# Running the model on xcore host interpreter with sample input image
# Sample image of a lion (ImageNet class 291)
with open("lion.bin", "rb") as f:
    data = f.read()

input_array = np.frombuffer(data, dtype=np.uint8).astype(np.int32)
# input image values are in the range 0 to 255
# we subtract 128 to change to -128 to 127 for int8
input_array = input_array - 128


interpreter = xcore_tflm_host_interpreter()
interpreter.set_model(model_path=opt_model_path, params_path=opt_params_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_data = np.asarray(input_array, dtype=interpreter.get_input_details()[0]["dtype"])
input_data = np.reshape(input_data, interpreter.get_input_details()[0]["shape"])
interpreter.set_tensor(input_details[0]["index"], input_data)
# Inference
interpreter.invoke()
detections = interpreter.get_tensor(output_details[0]["index"])
print("Inferred imagenet class = %d" % detections.argmax())
