import os
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "False")

from xmos_ai_tools import xformer
from ultralytics import YOLO
import numpy as np

try:
    from xmos_ai_tools.xinterpreters import TFLMHostInterpreter
except ImportError as exc:
    if "opcode2name" not in str(exc):
        raise
    import tflite
    from tflite.utils import opcode2name

    sys.modules.pop("xmos_ai_tools.xinterpreters.host_interpreter", None)
    sys.modules.pop("xmos_ai_tools.xinterpreters", None)
    tflite.opcode2name = opcode2name
    from xmos_ai_tools.xinterpreters import TFLMHostInterpreter

HEIGHT, WIDTH = 160, 160
TFLITE_MODEL_PATH = "yolov8n-cls_saved_model/yolov8n-cls_full_integer_quant.tflite"
OPT_MODEL_PATH = "src/model.tflite"
OPT_PARAMS_PATH = "src/model_flash.params"
NAMING_PREFIX = "model_"
SAMPLE_IMAGE_PATH = "lion.bin"


def _check_installed_dependencies():
    protobuf_version = version("protobuf")
    if protobuf_version != "4.25.5":
        raise RuntimeError(
            "This example requires protobuf==4.25.5 for onnx2tf. "
            f"Found protobuf=={protobuf_version}; run `python -m pip install -r requirements.txt` before rerunning."
        )


def _tensor_type_name(tensor_type):
    from tflite.TensorType import TensorType

    for name, value in TensorType.__dict__.items():
        if name.isupper() and value == tensor_type:
            return name
    return f"UNKNOWN({tensor_type})"


def _check_tflite_micro_compatible(tflite_model_path):
    from tflite.BuiltinOperator import BuiltinOperator
    from tflite.Model import Model
    from tflite.TensorType import TensorType

    model = Model.GetRootAsModel(bytearray(Path(tflite_model_path).read_bytes()), 0)
    failures = []

    for subgraph_index in range(model.SubgraphsLength()):
        subgraph = model.Subgraphs(subgraph_index)
        for operator_index in range(subgraph.OperatorsLength()):
            operator = subgraph.Operators(operator_index)
            operator_code = model.OperatorCodes(operator.OpcodeIndex()).BuiltinCode()
            if operator_code != BuiltinOperator.CONV_2D:
                continue

            checks = [
                ("input", operator.Inputs(0), {TensorType.INT8}),
                ("filter", operator.Inputs(1), {TensorType.INT8}),
                ("output", operator.Outputs(0), {TensorType.INT8}),
            ]
            if operator.InputsLength() > 2 and operator.Inputs(2) >= 0:
                checks.append(("bias", operator.Inputs(2), {TensorType.INT32}))

            for role, tensor_index, expected_types in checks:
                tensor_type = subgraph.Tensors(tensor_index).Type()
                if tensor_type not in expected_types:
                    expected = ", ".join(_tensor_type_name(value) for value in expected_types)
                    failures.append(
                        f"subgraph {subgraph_index} operator {operator_index} CONV_2D {role} "
                        f"tensor is {_tensor_type_name(tensor_type)}, expected {expected}"
                    )

    if failures:
        details = "\n".join(f"  - {failure}" for failure in failures[:10])
        raise RuntimeError(
            "The exported TFLite model contains hybrid Conv2D tensors, which TFLite Micro cannot compile.\n"
            f"{details}"
        )


def _write_calibration_data(calibration_data_path):
    data = Path(SAMPLE_IMAGE_PATH).read_bytes()
    image = np.frombuffer(data, dtype=np.uint8).reshape(1, HEIGHT, WIDTH, 3).astype(np.float32)
    np.save(calibration_data_path, image)


def _convert_onnx_to_int8_tflite(onnx_model_path):
    import onnx2tf

    output_folder = Path(TFLITE_MODEL_PATH).parent
    output_folder.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as calibration_data_file:
        calibration_data_path = calibration_data_file.name
        try:
            _write_calibration_data(calibration_data_path)
            onnx2tf.convert(
                input_onnx_file_path=onnx_model_path,
                output_folder_path=str(output_folder),
                output_integer_quantized_tflite=True,
                custom_input_op_name_np_data_path=[
                    ["images", calibration_data_path, [[[[0, 0, 0]]]], [[[[255, 255, 255]]]]]
                ],
                input_quant_dtype="int8",
                output_quant_dtype="int8",
                quant_type="per-channel",
                tflite_backend="tf_converter",
                not_use_onnxsim=True,
                verbosity="error",
            )
        finally:
            Path(calibration_data_path).unlink(missing_ok=True)
    if not Path(TFLITE_MODEL_PATH).is_file() or Path(TFLITE_MODEL_PATH).stat().st_size == 0:
        raise RuntimeError(f"Expected full-int8 TFLite model was not generated: {TFLITE_MODEL_PATH}")

###############################################
# Creating and converting an YoloV8 cls model #
###############################################

# Load a model
_check_installed_dependencies()
model = YOLO("yolov8n-cls.pt")  # load an official model

# Export the model
ONNX_MODEL_PATH = model.export(format="onnx", imgsz=(HEIGHT, WIDTH), opset=20, simplify=True)
_convert_onnx_to_int8_tflite(ONNX_MODEL_PATH)
_check_tflite_micro_compatible(TFLITE_MODEL_PATH)

# Convert the model to XCore optimized TFLite via xformer:
# There are various ways to configure the compiler to optimize the model,
# operator splitting isn't documented yet. This configuration works well for
# MobileNetV2, reach out if you need assistance with other complex models
xformer.convert(
    TFLITE_MODEL_PATH,
    OPT_MODEL_PATH,
    [
        ("xcore-weights-file", OPT_PARAMS_PATH),
        ("xcore-thread-count", "5"),
        ("xcore-naming-prefix", NAMING_PREFIX),
    ],
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
with open(SAMPLE_IMAGE_PATH, "rb") as f:
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
