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

def load_tests(name):
    if name.startswith('argmax'):
        pattern = os.path.join(directories.OP_TEST_MODELS_DATA_DIR,
            operator_codes.XCOREOpCodes.XC_argmax_16.name)
    elif name.startswith('conv2d_1x1'):
        pattern = os.path.join(directories.OP_TEST_MODELS_DATA_DIR,
            operator_codes.XCOREOpCodes.XC_conv2d_1x1.name)
    elif name.startswith('conv2d_shallowin_deepout'):
        pattern = os.path.join(directories.OP_TEST_MODELS_DATA_DIR,
            operator_codes.XCOREOpCodes.XC_conv2d_shallowin_deepout_relu.name)
    elif name.startswith('conv2d_deepin_deepout'):
        pattern = os.path.join(directories.OP_TEST_MODELS_DATA_DIR,
            operator_codes.XCOREOpCodes.XC_conv2d_deepin_deepout_relu.name)
    elif name.startswith('test_fully_connected'):
        pattern = os.path.join(directories.OP_TEST_MODELS_DATA_DIR,
            operator_codes.XCOREOpCodes.XC_fc_deepin_anyout.name)
    elif name.startswith('maxpool2d'):
        pattern = os.path.join(directories.OP_TEST_MODELS_DATA_DIR,
            operator_codes.XCOREOpCodes.XC_maxpool2d.name)
    elif name.startswith('avgpool2d_global'):
        pattern = os.path.join(directories.OP_TEST_MODELS_DATA_DIR,
            operator_codes.XCOREOpCodes.XC_avgpool2d_global.name)
    elif name.startswith('avgpool2d'):
        pattern = os.path.join(directories.OP_TEST_MODELS_DATA_DIR,
            operator_codes.XCOREOpCodes.XC_avgpool2d.name)
    elif name.startswith('requantize_16_8'):
        name = f'{operator_codes.XCOREOpCodes.XC_requantize_16_to_8.name}'
        pattern = os.path.join(directories.OP_TEST_MODELS_DATA_DIR, name)
    elif name.startswith('lookup_8'):
        name = f'{operator_codes.XCOREOpCodes.XC_lookup_8.name}'
        pattern = os.path.join(directories.OP_TEST_MODELS_DATA_DIR, name)
    else:
        raise Exception(f'Unsupported op model: {name}')

    test_cases = []

    for directory in Path(pattern).rglob('*'):
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
    test_file = metafunc.config.getoption('--test-file')

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
        cmd = f'{test_model_app}q {flatbuffer} {input_file} {predicted_output_file}'
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

def test_lookup(test_model_app, lookup_8_test_case):
    assert(run_test_case(test_model_app, lookup_8_test_case))


def test_argmax(test_model_app, argmax_test_case):
    assert(run_test_case(test_model_app, argmax_test_case))


def test_conv2d_1x1(test_model_app, conv2d_1x1_test_case):
    assert(run_test_case(test_model_app, conv2d_1x1_test_case))


@pytest.mark.xfail
def test_conv2d_shallowin_deepout(test_model_app, conv2d_shallowin_deepout_test_case):
    assert(run_test_case(test_model_app, conv2d_shallowin_deepout_test_case))


@pytest.mark.xfail
def test_conv2d_deepin_deepout(test_model_app, conv2d_deepin_deepout_test_case):
    assert(run_test_case(test_model_app, conv2d_deepin_deepout_test_case))


def test_fully_connected(test_model_app, test_fully_connected_test_case):
    assert(run_test_case(test_model_app, test_fully_connected_test_case))


def test_maxpool2d(test_model_app, maxpool2d_test_case):
    assert(run_test_case(test_model_app, maxpool2d_test_case))

@pytest.mark.xfail
def test_avgpool2d(test_model_app, avgpool2d_test_case):
    assert(run_test_case(test_model_app, avgpool2d_test_case))


def test_avgpool2d_global(test_model_app, avgpool2d_global_test_case):
    assert(run_test_case(test_model_app, avgpool2d_global_test_case))


def test_requantize(test_model_app, requantize_16_8_test_case):
    assert(run_test_case(test_model_app, requantize_16_8_test_case))


if __name__ == "__main__":
    pytest.main()
