# Copyright (c) 2019, XMOS Ltd, All rights reserved

import os
import sys
import json
import glob
from pathlib import Path
import subprocess
import tempfile
import pytest

import numpy as np

import helpers
import directories
from tflite2xcore import operator_codes
from tflite2xcore.serialization import read_flatbuffer
from tflite2xcore.xcore_interpreter import XCOREInterpreter
from tflite2xcore.xcore_model import TensorType


def load_tests(test_name, test_dir, max_count):
    supported_operators = set([
        operator_codes.XCOREOpCodes.XC_argmax_16.name,
        operator_codes.XCOREOpCodes.XC_conv2d_1x1.name,
        operator_codes.XCOREOpCodes.XC_conv2d_depthwise.name,
        operator_codes.XCOREOpCodes.XC_conv2d_shallowin_deepout_relu.name,
        operator_codes.XCOREOpCodes.XC_conv2d_deep.name,
        operator_codes.XCOREOpCodes.XC_fc_deepin_anyout.name,
        operator_codes.XCOREOpCodes.XC_maxpool2d.name,
        operator_codes.XCOREOpCodes.XC_avgpool2d.name,
        operator_codes.XCOREOpCodes.XC_avgpool2d_global.name,
        operator_codes.XCOREOpCodes.XC_lookup_8.name,
        operator_codes.XCOREOpCodes.XC_requantize_16_to_8.name
    ])

    operator_name = test_name[:-10]
    if operator_name not in supported_operators:
        raise Exception(f'Unsupported op model: {operator_name}')

    test_cases = []
    for directory in Path(os.path.join(test_dir, operator_name)).rglob('*'):
        if os.path.isdir(directory):
            flatbuffer_xcore = os.path.join(directory, 'models/model_xcore.tflite')
            if os.path.isfile(flatbuffer_xcore):
                input_files = glob.glob(os.path.join(directory, 'test_data/model_xcore/*.x'))
                model = read_flatbuffer(flatbuffer_xcore)
                input_quantization = model.subgraphs[0].outputs[0].quantization

                flatbuffer_stripped = os.path.join(directory, 'models/model_stripped.tflite')
                output_files = glob.glob(os.path.join(directory, 'test_data/model_stripped/*.y'))
                model = read_flatbuffer(flatbuffer_stripped)
                output_quantization = model.subgraphs[0].outputs[0].quantization

                for input_file, output_file in zip(sorted(input_files), sorted(output_files)):
                    test_cases.append({
                        'id': '/'.join(directory.parts[-2:])+'_',
                        'flatbuffer': flatbuffer_xcore,
                        'input': {
                            'filename': input_file,
                            'quantization': input_quantization
                        },
                        'expected_output': {
                            'filename': output_file,
                            'quantization': output_quantization
                        }
                    })
                    if len(test_cases) >= max_count:
                        break 

    return test_cases


def pytest_generate_tests(metafunc):
    max_count = metafunc.config.getoption('--max-count')
    test_dir = metafunc.config.getoption('--test-dir')

    for fixture in metafunc.fixturenames:
        if fixture.endswith('test_case'):
            tests = load_tests(fixture, test_dir, max_count)
            ids = [test['id'] for test in tests]
            metafunc.parametrize(fixture, tests, ids=ids)


def run_test_case(test_model_app, test_case, abs_tol=1):
    flatbuffer = test_case['flatbuffer']
    input_file = test_case['input']['filename']
    predicted_quantization = test_case['input']['quantization']
    expected_output_file = test_case['expected_output']['filename']
    expected_quantization = test_case['expected_output']['quantization']
    predicted_output_file = os.path.join(tempfile.mkdtemp(), 'predicted_output.bin')

    print('**********')
    print('* Inputs *')
    print('**********')
    # print('Testcase:', json.dumps(test_case, indent=4))
    print('***********')
    print('* Results *')
    print('***********')
    if test_model_app:
        if test_model_app.endswith('.xe'):
            cmd = f'xsim --args {test_model_app} {flatbuffer} {input_file} {predicted_output_file}'
        else:
            cmd = f'{test_model_app} {flatbuffer} {input_file} {predicted_output_file}'
        print('Command:', cmd)
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
            result = helpers.compare_tensor_files(expected_output_file, expected_quantization,
                predicted_output_file, predicted_quantization, abs_tol)

            os.remove(predicted_output_file)

            return result
        except subprocess.CalledProcessError as ex:
            print(ex.output.decode('utf-8'))
            print(ex)
            return False
    else:
        # use interpreter
        model = read_flatbuffer(flatbuffer)

        input_tensor = model.subgraphs[0].inputs[0]
        input_index= model.subgraphs[0].tensors.index(model.subgraphs[0].inputs[0])
        output_tensor = model.subgraphs[0].outputs[0]
        output_index= model.subgraphs[0].tensors.index(model.subgraphs[0].outputs[0])

        with open(flatbuffer, 'rb') as fd:
            model_content = fd.read()

        interpreter = XCOREInterpreter(model_content=model_content)
        interpreter.allocate_tensors()

        input_ = np.fromfile(input_file, dtype=TensorType.to_numpy_dtype(input_tensor.type))
        input_ = input_.reshape(input_tensor.shape)

        interpreter.set_tensor(input_index, input_)
        interpreter.invoke()
        predicted_output = interpreter.get_tensor(output_index)

        with open(predicted_output_file, 'wb') as fd:
            fd.write(predicted_output.tobytes())

        result = helpers.compare_tensor_files(expected_output_file, expected_quantization,
            predicted_output_file, predicted_quantization, abs_tol)

        return result

def is_xfail(test_case):
    index = test_case['id'].find('xfail=true')
    return index > 0

def test_XC_lookup_8(test_model_app, XC_lookup_8_test_case, abs_tol):
    if is_xfail(XC_lookup_8_test_case):
        pytest.xfail()
    assert(run_test_case(test_model_app, XC_lookup_8_test_case, abs_tol))


def test_XC_argmax_16(test_model_app, XC_argmax_16_test_case, abs_tol):
    if is_xfail(XC_argmax_16_test_case):
        pytest.xfail()
    assert(run_test_case(test_model_app, XC_argmax_16_test_case, abs_tol))


def test_XC_conv2d_1x1(test_model_app, XC_conv2d_1x1_test_case, abs_tol):
    if is_xfail(XC_conv2d_1x1_test_case):
        pytest.xfail()
    assert(run_test_case(test_model_app, XC_conv2d_1x1_test_case, abs_tol))


def test_XC_conv2d_depthwise(test_model_app, XC_conv2d_depthwise_test_case, abs_tol):
    if is_xfail(XC_conv2d_depthwise_test_case):
        pytest.xfail()
    assert(run_test_case(test_model_app, XC_conv2d_depthwise_test_case, abs_tol))


def test_XC_conv2d_shallowin_deepout_relu(test_model_app, XC_conv2d_shallowin_deepout_relu_test_case, abs_tol):
    if is_xfail(XC_conv2d_shallowin_deepout_relu_test_case):
        pytest.xfail()
    assert(run_test_case(test_model_app, XC_conv2d_shallowin_deepout_relu_test_case, abs_tol))


def test_XC_conv2d_deep(test_model_app, XC_conv2d_deep_test_case, abs_tol):
    if is_xfail(XC_conv2d_deep_test_case):
        pytest.xfail()
    assert(run_test_case(test_model_app, XC_conv2d_deep_test_case, abs_tol))


def test_XC_fc_deepin_anyout(test_model_app, XC_fc_deepin_anyout_test_case, abs_tol):
    if is_xfail(XC_fc_deepin_anyout_test_case):
        pytest.xfail()
    assert(run_test_case(test_model_app, XC_fc_deepin_anyout_test_case, abs_tol))


def test_XC_maxpool2d(test_model_app, XC_maxpool2d_test_case, abs_tol):
    if is_xfail(XC_maxpool2d_test_case):
        pytest.xfail()
    assert(run_test_case(test_model_app, XC_maxpool2d_test_case, abs_tol))


def test_XC_avgpool2d(test_model_app, XC_avgpool2d_test_case, abs_tol):
    if is_xfail(XC_avgpool2d_test_case):
        pytest.xfail()
    assert(run_test_case(test_model_app, XC_avgpool2d_test_case, abs_tol))


def test_XC_avgpool2d_global(test_model_app, XC_avgpool2d_global_test_case, abs_tol):
    if is_xfail(XC_avgpool2d_global_test_case):
        pytest.xfail()
    assert(run_test_case(test_model_app, XC_avgpool2d_global_test_case, abs_tol))


def test_XC_requantize_16_to_8(test_model_app, XC_requantize_16_to_8_test_case, abs_tol):
    if is_xfail(XC_requantize_16_to_8_test_case):
        pytest.xfail()
    assert(run_test_case(test_model_app, XC_requantize_16_to_8_test_case, abs_tol))


if __name__ == "__main__":
    pytest.main()
