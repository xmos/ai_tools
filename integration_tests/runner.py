import pathlib
import logging
import tempfile
from _pytest.fixtures import FixtureRequest
import numpy as np
import os
import subprocess
import larq_compute_engine as lce
import tensorflow as tf
from xtflm_interpreter import XTFLMInterpreter

# This error tolerance works for the models we have currently
# The maximum error we see is 1.037735
ABSOLUTE_ERROR_TOLERANCE = 1.04
LOGGER = logging.getLogger(__name__)
XFORMER2_PATH = (
    pathlib.Path(__file__)
    .resolve()
    .parent.parent.joinpath("experimental", "xformer", "bazel-bin", "xcore-opt")
)


def dequantize(arr: np.ndarray, scale: float, zero_point: int) -> np.float32:
    return np.float32(arr.astype(np.int32) - np.int32(zero_point)) * np.float32(scale)


def get_xformed_model(model: bytes) -> bytes:
    # write input model to temporary file
    input_file = tempfile.NamedTemporaryFile(delete=False)
    input_file.write(model)
    input_file.close()
    # create another temp file for output model
    output_file = tempfile.NamedTemporaryFile(delete=False)
    cmd = [
        str(XFORMER2_PATH),
        str(input_file.name),
        "-o",
        str(output_file.name),
        "--xcore-thread-count=5",
        # "--xcore-replace-avgpool-with-conv2d",
        # "--xcore-replace-with-conv2dv2",
        # "--xcore-translate-to-customop"
    ]
    p = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True
    )
    LOGGER.info(p.stdout)

    # read output model content
    bits = bytes(output_file.read())
    output_file.close()
    os.remove(input_file.name)
    os.remove(output_file.name)
    return bits


# Run the model on Larq/TFLite interpreter and compare the output with xformed model on XCore TFLM
def test_model(request: FixtureRequest, filename: str) -> None:
    if not XFORMER2_PATH.exists():
        LOGGER.error(
            "xcore-opt not found! Please build xformer before running integration tests!"
        )
        assert False
    model_path = pathlib.Path(filename).resolve()
    if not model_path.exists():
        LOGGER.error("model file not found!")
        assert False
    # read in model from file
    with open(model_path, "rb") as fd:
        model_content = fd.read()

    testing_binary_models = request.config.getoption("bnn")
    if testing_binary_models:
        LOGGER.info("Creating LCE interpreter...")
        interpreter = lce.testing.Interpreter(
            model_content, num_threads=1, use_reference_bconv=True
        )
        # interpreter = lce.testing.Interpreter(model_content, num_threads=1)
        input_tensor_type = interpreter.input_types[0]
        input_tensor_shape = interpreter.input_shapes[0]
    else:
        LOGGER.info("Creating TFLite interpreter...")
        interpreter = tf.lite.Interpreter(
            model_content=model_content,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
            experimental_preserve_all_tensors=True,
        )
        # interpreter = tensorflow.lite.Interpreter(
        #     model_content=model_content)
        interpreter.allocate_tensors()
        input_tensor_details = interpreter.get_input_details()[0]
        input_tensor_type = input_tensor_details["dtype"]
        input_tensor_shape = input_tensor_details["shape"]

    LOGGER.info("Invoking xformer to get xformed model...")
    xformed_model = get_xformed_model(model_content)
    LOGGER.info("Creating TFLM XCore interpreter...")
    ie = XTFLMInterpreter(model_content=xformed_model)

    # Run tests
    num_of_fails = 0
    number_of_samples = request.config.getoption("number_of_samples")
    for test in range(0, int(number_of_samples)):
        LOGGER.info("Run #" + str(test))
        LOGGER.info("Creating random input...")
        # input_tensor = np.array(
        #     np.random.uniform(-1, 1, input_tensor_shape), dtype=input_tensor_type
        # )
        input_tensor = np.array(
            100 * np.random.random_sample(input_tensor_shape), dtype=input_tensor_type
        )
        # input_tensor = np.array(
        #    100 * np.ones(input_tensor_shape), dtype=input_tensor_type
        # )

        if testing_binary_models:
            LOGGER.info("Invoking LCE interpreter...")
            outputs = interpreter.predict(input_tensor)
            # for some reason, batch dim is missing in lce when only one output
            if len(outputs) == 1:
                outputs = [outputs]
            num_of_outputs = len(outputs)
            output_scales = interpreter.output_scales
            output_zero_points = interpreter.output_zero_points
        else:
            interpreter.set_tensor(input_tensor_details["index"], input_tensor)
            LOGGER.info("Invoking TFLite interpreter...")
            interpreter.invoke()
            num_of_outputs = len(interpreter.get_output_details())
            outputs = []
            output_scales = []
            output_zero_points = []
            for i in range(num_of_outputs):
                outputs.append(
                    interpreter.get_tensor(interpreter.get_output_details()[i]["index"])
                )
                quant_params = interpreter.get_output_details()[i][
                    "quantization_parameters"
                ]
                output_scales.append(quant_params["scales"])
                output_zero_points.append(quant_params["zero_points"])

        LOGGER.info("Invoking XCORE interpreter...")
        ie.set_input_tensor(0, input_tensor)
        ie.invoke()
        xformer_outputs = []
        for i in range(num_of_outputs):
            xformer_outputs.append(ie.get_output_tensor(i))

        # Compare outputs
        for i in range(num_of_outputs):
            LOGGER.info("Comparing output number " + str(i) + "...")
            # if quantized output, we dequantize it before comparing
            if output_scales[i]:
                outputs[i] = dequantize(
                    outputs[i], output_scales[i], output_zero_points[i]
                )
                xformer_outputs[i] = dequantize(
                    xformer_outputs[i], output_scales[i], output_zero_points[i]
                )
            np.set_printoptions(threshold=np.inf)
            LOGGER.debug("xformer output :\n{0}".format(xformer_outputs[i]))
            LOGGER.debug("compared output :\n{0}".format(outputs[i]))
            try:
                np.testing.assert_allclose(
                    xformer_outputs[i],
                    outputs[i],
                    atol=ABSOLUTE_ERROR_TOLERANCE,
                )
            except Exception as e:
                num_of_fails += 1
                LOGGER.error(e)
                d = ~np.isclose(
                    outputs[i],
                    xformer_outputs[i],
                    atol=ABSOLUTE_ERROR_TOLERANCE,
                )
                LOGGER.error(
                    "Mismatched element indices :\n{0}".format(np.flatnonzero(d))
                )
                LOGGER.error(
                    "Mismatched elements from xformer output :\n{0}".format(
                        xformer_outputs[i][d]
                    )
                )
                LOGGER.error(
                    "Mismatched elements from compared output :\n{0}".format(
                        outputs[i][d]
                    )
                )
                LOGGER.error("Run #" + str(test) + " failed")
    assert num_of_fails == 0
