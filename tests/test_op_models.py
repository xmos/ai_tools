# Copyright (c) 2019, XMOS Ltd, All rights reserved

import os
import sys
import json
import glob
from pathlib import Path
import subprocess
import tempfile
import pytest

import helpers
import directories
from tflite2xcore import read_flatbuffer
from tflite2xcore import operator_codes

def load_tests(test_name):
    supported_operators = set([
        operator_codes.XCOREOpCodes.XC_argmax_16.name,
        operator_codes.XCOREOpCodes.XC_conv2d_1x1.name,
        operator_codes.XCOREOpCodes.XC_conv2d_depthwise.name,
        operator_codes.XCOREOpCodes.XC_conv2d_shallowin_deepout_relu.name,
        operator_codes.XCOREOpCodes.XC_conv2d_deepin_deepout_relu.name,
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
    for directory in Path(os.path.join(directories.OP_TEST_MODELS_DATA_DIR, operator_name)).rglob('*'):
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
                        'id': '/'.join(directory.parts[-2:]),
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

    return test_cases


def pytest_generate_tests(metafunc):
    for fixture in metafunc.fixturenames:
        if fixture.endswith('test_case'):
            tests = load_tests(fixture)
            ids = [test['id'] for test in tests]
            metafunc.parametrize(fixture, tests, ids=ids)


def run_test_case(test_model_app, test_case, abs_tol=1):
    flatbuffer = test_case['flatbuffer']
    input_file = test_case['input']['filename']
    predicted_quantization = test_case['input']['quantization']
    expected_output_file = test_case['expected_output']['filename']
    expected_quantization = test_case['expected_output']['quantization']
    predicted_output_file = os.path.join(tempfile.mkdtemp(), 'predicted_output.bin')
    if test_model_app.endswith('.xe'):
        cmd = f'xsim --args {test_model_app} {flatbuffer} {input_file} {predicted_output_file}'
    else:
        cmd = f'{test_model_app} {flatbuffer} {input_file} {predicted_output_file}'
    print('**********')
    print('* Inputs *')
    print('**********')
    print('Testcase:', json.dumps(test_case, indent=4))
    print('Command:', cmd)
    print('***********')
    print('* Results *')
    print('***********')
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        result = helpers.compare_tensor_files(expected_output_file, expected_quantization,
            predicted_output_file, predicted_quantization, abs_tol)

        os.remove(predicted_output_file)

        return result
    except subprocess.CalledProcessError as ex:
        print(ex)
        return False

def test_XC_lookup_8(test_model_app, XC_lookup_8_test_case):
    assert(run_test_case(test_model_app, XC_lookup_8_test_case))


def test_XC_argmax_16(test_model_app, XC_argmax_16_test_case):
    assert(run_test_case(test_model_app, XC_argmax_16_test_case))


def test_XC_conv2d_1x1(test_model_app, XC_conv2d_1x1_test_case):
    assert(run_test_case(test_model_app, XC_conv2d_1x1_test_case))


def test_XC_conv2d_depthwise(test_model_app, XC_conv2d_depthwise_test_case):
    assert(run_test_case(test_model_app, XC_conv2d_depthwise_test_case))


@pytest.mark.xfail
def test_XC_conv2d_shallowin_deepout_relu(test_model_app, XC_conv2d_shallowin_deepout_relu_test_case):
    assert(run_test_case(test_model_app, XC_conv2d_shallowin_deepout_relu_test_case))


@pytest.mark.xfail
def test_XC_conv2d_deepin_deepout_relu(test_model_app, XC_conv2d_deepin_deepout_relu_test_case):
    assert(run_test_case(test_model_app, XC_conv2d_deepin_deepout_relu_test_case))


def test_XC_fc_deepin_anyout(test_model_app, XC_fc_deepin_anyout_test_case):
    assert(run_test_case(test_model_app, XC_fc_deepin_anyout_test_case))


def test_XC_maxpool2d(test_model_app, XC_maxpool2d_test_case):
    assert(run_test_case(test_model_app, XC_maxpool2d_test_case))


@pytest.mark.xfail
def test_XC_avgpool2d(test_model_app, XC_avgpool2d_test_case):
    assert(run_test_case(test_model_app, XC_avgpool2d_test_case))


def test_XC_avgpool2d_global(test_model_app, XC_avgpool2d_global_test_case):
    assert(run_test_case(test_model_app, XC_avgpool2d_global_test_case))


def test_XC_requantize_16_to_8(test_model_app, XC_requantize_16_to_8_test_case):
    assert(run_test_case(test_model_app, XC_requantize_16_to_8_test_case))


if __name__ == "__main__":
    pytest.main()
