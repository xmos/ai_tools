import pathlib
import logging
import tempfile
from _pytest.fixtures import FixtureRequest
import numpy as np
import os
import sys
import subprocess
import larq_compute_engine as lce
import tensorflow as tf
from xmos_ai_tools.xinterpreters import (
    xcore_tflm_host_interpreter,
    xcore_tflm_usb_interpreter,
)
from xmos_ai_tools import xformer
from itertools import chain
import yaml

MAX_ABS_ERROR = 1
ABS_AVG_ERROR = 1./4
AVG_ABS_ERROR = 1./4
REQUIRED_OUTPUTS = 2048
LOGGER = logging.getLogger(__name__)

LIB_TFLM_DIR_PATH = (pathlib.Path(__file__).resolve().parents[1] / "third_party" / "lib_tflite_micro")
TFLM_INCLUDE_PATH = pathlib.Path.joinpath(LIB_TFLM_DIR_PATH, "lib_tflite_micro", "submodules", "tflite-micro")
FLATBUFFERS_INCLUDE_PATH = pathlib.Path.joinpath(LIB_TFLM_DIR_PATH, "lib_tflite_micro", "submodules", "flatbuffers", "include")
TFLMC_DIR_PATH = pathlib.Path.joinpath(LIB_TFLM_DIR_PATH, "tflite_micro_compiler")
TFLMC_BUILD_DIR_PATH = pathlib.Path.joinpath(TFLMC_DIR_PATH, "build")
TFLMC_EXE_PATH = pathlib.Path.joinpath(TFLMC_BUILD_DIR_PATH, "tflite_micro_compiler")
TFLMC_MAIN_CPP_PATH = pathlib.Path.joinpath(TFLMC_DIR_PATH, "model_main.cpp")

def run_cmd(cmd, working_dir = None):
    try:
        if working_dir:
            p = subprocess.run(cmd,
                            cwd = working_dir,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            check=True)
        else:
            p = subprocess.run(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            check=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        sys.exit(1)

def get_tflmc_model_exe(model, dirname):
    # write model to temp directory
    model_path = pathlib.Path(dirname) / "input.tflite"
    with open(pathlib.Path(model_path).resolve(), "wb") as fd:
        fd.write(model)

    # compile model with tflmc and create model.cpp
    model_cpp_path = pathlib.Path(dirname) / "model.cpp"
    cmd = [str(TFLMC_EXE_PATH), str(model_path), str(model_cpp_path)]
    run_cmd(cmd)

    # compile model.cpp with model_main.cpp
    model_exe_path = pathlib.Path(dirname) / "a.out"
    cmd = ["clang++",
    "-DTF_LITE_DISABLE_X86_NEON",
    "-DTF_LITE_STATIC_MEMORY",
    "-DNO_INTERPRETER",
    "-std=c++14",
    "-I" + str(TFLM_INCLUDE_PATH),
    "-I" + str(FLATBUFFERS_INCLUDE_PATH),
    "-I" + dirname,
    "-I" + os.getenv("CONDA_PREFIX") + "/include",
    "-g",
    "-O0",
    "-lxtflitemicro",
    "-L" + str(TFLMC_BUILD_DIR_PATH),
    str(model_cpp_path),
    str(TFLMC_MAIN_CPP_PATH),
    "-rpath",
    str(TFLMC_BUILD_DIR_PATH),
    "-o",
    str(model_exe_path)
    ]
    print(" ".join(cmd))
    run_cmd(cmd)
    return model_exe_path

def get_tflmc_outputs(model_exe_path, input_tensor, tfl_outputs):
    dirname = model_exe_path.parents[0]
    # write input tensor to npy in temp directory
    input_npy = dirname / "in.npy"
    np.save(input_npy, input_tensor)
    # run a.out with in.npy
    cmd = [str(model_exe_path), str(input_npy)]
    run_cmd(cmd, dirname)

    # read in out.npy into xformer_outputs
    xformer_outputs = []
    for i in range(len(tfl_outputs)):
        output_filename = str(i) + ".npy"
        output_arr = np.load(dirname / output_filename)
        # print(tfl_outputs[0].shape)
        reshaped_arr = np.reshape(output_arr, tfl_outputs[i].shape)
        # print(reshaped_arr)
        xformer_outputs.append(reshaped_arr)

    return xformer_outputs

def get_xformed_model(model: bytes) -> bytes:
    # write input model to temporary file
    input_file = tempfile.NamedTemporaryFile(delete=False)
    input_file.write(model)
    input_file.close()
    # create another temp file for output model
    output_file = tempfile.NamedTemporaryFile(delete=False)

    xformer.convert(str(input_file.name), str(output_file.name), {
    "xcore-thread-count": 5,
})

    # read output model content
    bits = bytes(output_file.read())
    output_file.close()
    os.remove(input_file.name)
    os.remove(output_file.name)
    return bits


# Run the model on Larq/TFLite interpreter and compare the output with xformed model on XCore TFLM
def test_model(request: FixtureRequest, filename: str) -> None:
    # for attaching a debugger
    if request.config.getoption("s"):
        import time
        time.sleep(5)

    # read options passed in
    testing_binary_models_option = request.config.getoption("bnn")
    testing_device_option = request.config.getoption("device")
    testing_on_tflmc_option = request.config.getoption("tflmc")
    number_of_samples_option = request.config.getoption("number_of_samples")
    testing_detection_postprocess_option = True if "detection_postprocess" in request.node.name else False

    params = dict()
    params['MAX_ABS_ERROR'] = MAX_ABS_ERROR
    params['ABS_AVG_ERROR'] = ABS_AVG_ERROR
    params['AVG_ABS_ERROR'] = AVG_ABS_ERROR
    params['REQUIRED_OUTPUTS'] = REQUIRED_OUTPUTS

    model_path = pathlib.Path(filename).resolve()

    yaml_filename = 'params.yaml'

    yaml_filename = os.path.join(os.path.dirname(model_path), yaml_filename)
    
    if os.path.exists(yaml_filename):
        with open (yaml_filename) as f:
            yaml_params = yaml.safe_load(f)
            params.update(yaml_params)

    if not model_path.exists():
        LOGGER.error("model file not found!")
        assert False
    # read in model from file
    with open(model_path, "rb") as fd:
        model_content = fd.read()

    if testing_binary_models_option:
        LOGGER.info("Creating LCE interpreter...")
        interpreter = lce.testing.Interpreter(
            model_content, num_threads=1, use_reference_bconv=True
        )
        # interpreter = lce.testing.Interpreter(model_content, num_threads=1)
        num_of_inputs = len(interpreter.input_types)
        input_tensor_type = []
        input_tensor_shape = []
        for i in range(num_of_inputs):
            input_tensor_type.append(interpreter.input_types[i])
            input_tensor_shape.append(interpreter.input_shapes[i])
    else:
        LOGGER.info("Creating TFLite interpreter...")
        interpreter = tf.lite.Interpreter(
            model_content=model_content,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
            experimental_preserve_all_tensors=True,
        )
        # interpreter = tf.lite.Interpreter(
        #     model_content=model_content)
        interpreter.allocate_tensors()
        num_of_inputs = len(interpreter.get_input_details())
        input_tensor_type = []
        input_tensor_shape = []
        for i in range(num_of_inputs):
            input_tensor_type.append(interpreter.get_input_details()[i]["dtype"])
            input_tensor_shape.append(interpreter.get_input_details()[i]["shape"])

    if testing_on_tflmc_option:
        LOGGER.info("Creating tflmc model exe...")
        tflmc_temp_dirname = tempfile.TemporaryDirectory(suffix=str(os.getpid()))
        tflmc_model_exe = get_tflmc_model_exe(model_content, tflmc_temp_dirname.name)
    else:    
        LOGGER.info("Invoking xformer to get xformed model...")
        if testing_detection_postprocess_option:
            LOGGER.info("Detection postprocess special case - loading int8 model for xcore...")
            with open(model_path.parent.joinpath("test_dtp.xc"), "rb") as fd:
                model_content = fd.read()
        xformed_model = get_xformed_model(model_content)
        LOGGER.info("Creating TFLM XCore interpreter...")
        if testing_device_option:
            ie = xcore_tflm_usb_interpreter()
        else:
            ie = xcore_tflm_host_interpreter()
        ie.set_model(model_content=xformed_model, secondary_memory=False)


    # Run tests
    num_of_fails = 0

    running_output_count = 0
    running_output_error = 0
    running_output_abs_error = 0
    
    test = 0
    while running_output_count < params['REQUIRED_OUTPUTS']:
        LOGGER.info("Run #" + str(test))
        test += 1

        input_tensor = []
        if testing_detection_postprocess_option:
            LOGGER.info("Detection postprocess special case - loading input from files...")
            in1 = np.load(model_path.parent.joinpath("in1.npy"))
            input_tensor.append(in1)
            in2 = np.load(model_path.parent.joinpath("in2.npy"))
            input_tensor.append(in2)
        else:
            LOGGER.info("Creating random input...")
            for i in range(num_of_inputs):
                input_tensor.append(np.array(256 * np.random.random_sample(input_tensor_shape[i]) - 127, dtype=input_tensor_type[i]))
                #input_tensor.append(np.array(100 * np.ones(input_tensor_shape[i]), dtype=input_tensor_type[i]))

        if testing_binary_models_option:
            LOGGER.info("Invoking LCE interpreter...")
            outputs = interpreter.predict(input_tensor)
            # for some reason, batch dim is missing in lce when only one output
            if len(outputs) == 1:
                outputs = [outputs]
            num_of_outputs = len(outputs)
            output_scales = interpreter.output_scales
            output_zero_points = interpreter.output_zero_points
        else:
            for i in range(num_of_inputs):
                interpreter.set_tensor(interpreter.get_input_details()[i]["index"], input_tensor[i])
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

        if testing_on_tflmc_option:
            LOGGER.info("Invoking tflmc...")
            xformer_outputs = get_tflmc_outputs(tflmc_model_exe, input_tensor, outputs)
        else:
            LOGGER.info("Invoking XCORE interpreter...")
            for i in range(num_of_inputs):
                ie.set_tensor(i, input_tensor[i])
            ie.invoke()
            xformer_outputs = []
            for i in range(num_of_outputs):
                output_tensor = ie.get_tensor(ie.get_output_details()[i]["index"])
                LOGGER.info("outputs: " + str(output_tensor.shape))
                xformer_outputs.append(output_tensor)

        # Compare outputs
        errors = np.array(list(chain(*[np.array(a-b).reshape(-1) for a, b in zip(outputs, xformer_outputs)]))).reshape(-1)

        if len(errors) > 0:

            running_output_count += np.prod(errors.shape)
            running_output_error += np.sum(errors)
            running_output_abs_error += np.sum(np.abs(errors))
            
            max_abs_error = np.amax(np.abs(errors))

            if (max_abs_error > params['MAX_ABS_ERROR']):
                LOGGER.error("Max abs error is too high: " + str(max_abs_error))
                assert max_abs_error <= 1 

    np.set_printoptions(threshold=np.inf)
    avg_error = running_output_error / running_output_count
    avg_abs_error = running_output_abs_error / running_output_count
    LOGGER.info(str(max_abs_error) + ' ' + str(avg_error) +' ' +  str(avg_abs_error))
    
    failed = False
    if (abs(avg_error) > params['ABS_AVG_ERROR']):
        failed = True
        LOGGER.error("Abs avg error is too high: " + str(abs(avg_error)))

    if (avg_abs_error > params['AVG_ABS_ERROR']):
        failed = True
        LOGGER.error("Avg abs error is too high: " + str(avg_abs_error))
        
    if failed:
        num_of_fails+= 1
        LOGGER.error("Run #" + str(test) + " failed")
            
    if testing_on_tflmc_option:
       tflmc_temp_dirname.cleanup()
    else:
        # Free allocated objects and cleanup
        # For tflmc testing, we don't create xcore interpreter ie
        # Test comparison is done only on Tensorflow interpreter
        ie.close()
    assert num_of_fails == 0
