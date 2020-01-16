# Copyright (c) 2019, XMOS Ltd, All rights reserved

import os
import sys
import json
import glob
import subprocess
import tempfile
import pytest

import helpers

sys.path.append('../python/')
from tflite2xcore import read_flatbuffer


def load_tests(name):
    if name.startswith('argmax'):
        directory = 'data/single_op_models/argmax_16'
    elif name.startswith('conv_shallowin_deepout'):
        directory = 'data/single_op_models/conv2d_shallowin_deepout_relu'
    elif name.startswith('conv_deepin_deepout'):
        directory = 'data/single_op_models/conv2d_deepin_deepout_relu'
    elif name.startswith('fc_deepin_anyout_final'):
        directory = 'data/single_op_models/fc_deepin_shallowout_final'
    # elif name.startswith('fc_deepin_anyout_intermediate'):
    #     directory = 'data/single_op_models/{TBD}'
    elif name.startswith('maxpool'):
        directory = 'data/single_op_models/maxpool2d_deep'
    # elif name.startswith('avgpool'):
    #     directory = 'data/single_op_models/{TBD}'

    flatbuffer_xcore = os.path.join(directory, 'models/model_xcore.tflite')
    input_files = glob.glob(os.path.join(directory, 'test_data/model_xcore/*.x'))
    model = read_flatbuffer(flatbuffer_xcore)
    input_quantization = model.subgraphs[0].outputs[0].quantization

    flatbuffer_stripped = os.path.join(directory, 'models/model_stripped.tflite')
    output_files = glob.glob(os.path.join(directory, 'test_data/model_stripped/*.y'))
    model = read_flatbuffer(flatbuffer_stripped)
    output_quantization = model.subgraphs[0].outputs[0].quantization

    test_cases = []
    for input_file, output_file in zip(sorted(input_files), sorted(output_files)):
        test_cases.append({
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
            metafunc.parametrize(fixture, tests)


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
    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    return helpers.compare_tensor_files(expected_output_file, expected_quantization,
        predicted_output_file, predicted_quantization, abs_tol)


def test_argmax(test_model_app, argmax_test_case):
    assert(run_test_case(test_model_app, argmax_test_case))


def test_conv_shallowin_deepout(test_model_app, conv_shallowin_deepout_test_case):
    assert(run_test_case(test_model_app, test_conv_shallowin_deepout_test_case))


def test_conv_deepin_deepout(test_model_app, conv_deepin_deepout_test_case):
    assert(run_test_case(test_model_app, conv_deepin_deepout_test_case))


def test_fc_deepin_anyout_final(test_model_app, fc_deepin_anyout_final_test_case):
    assert(run_test_case(test_model_app, fc_deepin_anyout_final_test_case))


# def test_fc_deepin_anyout_intermediate(test_model_app, fc_deepin_anyout_intermediate_test_case):
#     assert(run_test_case(test_model_app, fc_deepin_anyout_intermediate_test_case))


def test_maxpool(test_model_app, maxpool_test_case):
    assert(run_test_case(test_model_app, maxpool_test_case))


# def test_avgpool(test_model_app, avgpool_test_case):
#     assert(run_test_case(test_model_app, avgpool_test_case))


if __name__ == "__main__":
    pytest.main()
