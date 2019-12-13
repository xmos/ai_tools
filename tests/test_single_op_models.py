# Copyright (c) 2019, XMOS Ltd, All rights reserved

import os
import sys
import glob
import subprocess
import tempfile
import pytest

import helpers

sys.path.append('/home/kmoulton/repos/hotdog/ai_tools/python/')
from tflite2xcore import read_flatbuffer

@pytest.fixture()
def test_dataset(request):
    def load_test_dataset(directory):
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
            test_cases.append(
                {
                    'input': {
                        'filename': input_file,
                        'quantization': input_quantization
                    },
                    'expected_output': {
                        'filename': output_file,
                        'quantization': output_quantization
                    }
                }
            )

        return {
            'flatbuffer': flatbuffer_xcore,
            'test_cases': test_cases
        }

    return load_test_dataset


def run_dataset(test_model_app, dataset):
    flatbuffer = dataset['flatbuffer']
    print('flatbuffer:', flatbuffer)
    for test_case in dataset['test_cases']:
        print('test_case:', test_case)
        input_file = test_case['input']['filename']
        predicted_quantization = test_case['input']['quantization']
        expected_output_file = test_case['expected_output']['filename']
        expected_quantization = test_case['expected_output']['quantization']
        predicted_output_file = os.path.join(tempfile.mkdtemp(), 'predicted_output.bin')
        if test_model_app.endswith('.xe'):
            cmd = f'xsim --args {test_model_app} {flatbuffer} {input_file} {predicted_output_file}'
        else:
            cmd = f'{test_model_app} {flatbuffer} {input_file} {predicted_output_file}'
        print('command:', cmd)
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        yield helpers.compare_tensor_files(expected_output_file, expected_quantization,
            predicted_output_file, predicted_quantization)


def test_argmax(test_model_app, test_dataset):
    dataset = test_dataset(directory='data/single_op_models/argmax_16')
    assert(all(run_dataset(test_model_app, dataset)))


def test_conv2d_scheme1(test_model_app, test_dataset):
    dataset = test_dataset(directory='data/single_op_models/conv2d_shallowin_deepout_relu')
    assert(all(run_dataset(test_model_app, dataset)))


def test_conv2d_scheme2(test_model_app, test_dataset):
    dataset = test_dataset(directory='data/single_op_models/conv2d_deepin_deepout_relu')
    assert(all(run_dataset(test_model_app, dataset)))


def test_fully_connected_scheme1(test_model_app, test_dataset):
    dataset = test_dataset(directory='data/single_op_models/fc_deepin_shallowout_final')
    assert(all(run_dataset(test_model_app, dataset)))


def test_maxpool(test_model_app, test_dataset):
    dataset = test_dataset(directory='data/single_op_models/maxpool2d_deep')
    assert(all(run_dataset(test_model_app, dataset)))


if __name__ == "__main__":
    pytest.main()

