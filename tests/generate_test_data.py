#!/usr/bin/env python

# Copyright (c) 2019, XMOS Ltd, All rights reserved

import os
import sys
import re
import shutil
from functools import partial
import subprocess
import multiprocessing
import argparse

import directories
from tflite2xcore import operator_codes

stdout_lock = multiprocessing.Lock()

def make_folder_and_arguments(**kwargs):
    folder_fields = []
    aurgment_fields = []
    for key, value in kwargs.items():
        hyphenless_key = re.search('(?<=-)\w+', key).group(0) # strip off leading hyphens
        if isinstance(value, tuple):
            value_folder_str = 'x'.join([str(v) for v in value])
            value_argument_str = ' '.join([str(v) for v in value])
        else:
            value_folder_str = str(value)
            value_argument_str = str(value)

        folder_fields.append(f'{hyphenless_key}={value_folder_str}')
        aurgment_fields.append(f'{key} {value_argument_str}')

    return '_'.join(folder_fields), ' '.join(aurgment_fields)

def generate_test_case(dry_run, test_case):
    if test_case['train_model']:
        train_model_flag = '--train_model'
    else:
        train_model_flag = ''

    parameters = test_case['parameters']
    operator = test_case['operator']
    generator = test_case['generator']

    folder, arguments = make_folder_and_arguments(**parameters)
    output_dir = os.path.join(directories.OP_TEST_MODELS_DATA_DIR, operator, folder)
    cmd = f'python {generator} {train_model_flag} {arguments} -path {output_dir}'
    with stdout_lock:
        print(f'generating test case {output_dir}')
        print(f'   command: {cmd}')
    if not dry_run:
        try:
            subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as cmdexc:                                                                                                   
            print(cmdexc.output.decode('utf-8'))

def create_test_cases(operator, generator, parameter_sets, *, train_model=False):
     test_cases = []
     
     for i, parameter_sets in enumerate(parameter_sets):
        test_cases.append({
            'operator': operator,
            'generator': generator,
            'train_model': train_model,
            'parameters': parameter_sets
        })

     return test_cases

def run_generate(tests, jobs):
    test_cases = []

    #***********************************
    # AvgPool
    #***********************************
    operator = operator_codes.XCOREOpCodes.XC_lookup_8.name
    generator = os.path.join(directories.GENERATOR_DIR, 'generate_lookup_8.py')
    parameter_sets = [
        {'-in': 32, '-hi': 5, '-wi': 5, '-act': 'relu', '--input_init': ('unif', -10, 10)},
        {'-in': 3, '-hi': 2, '-wi': 5, '-act': 'relu', '--input_init': ('unif', -10, -1)},
        {'-in': 4, '-hi': 5, '-wi': 2, '-act': 'relu', '--input_init': ('unif', 1, 10)},
        {'-in': 32, '-hi': 5, '-wi': 5, '-act': 'relu6', '--input_init': ('unif', -10, 7)},
        {'-in': 3, '-hi': 2, '-wi': 5, '-act': 'relu6', '--input_init': ('unif', 1, 2)},
        {'-in': 4, '-hi': 5, '-wi': 2, '-act': 'relu6', '--input_init': ('unif', 6, 10)},
        {'-in': 4, '-hi': 5, '-wi': 5, '-act': 'logistic', '--input_init': ('unif', -1, 1)},
        {'-in': 4, '-hi': 5, '-wi': 5, '-act': 'tanh', '--input_init': ('unif', -1, 1)}
    ]

    if operator in tests or len(tests) == 0:
        test_cases.extend(create_test_cases(operator, generator, parameter_sets))

    #***********************************
    # AvgPool
    #***********************************
    operator = operator_codes.XCOREOpCodes.XC_avgpool2d.name
    generator = os.path.join(directories.GENERATOR_DIR, 'generate_avgpool2d.py')
    parameter_sets = [
        {'-in': 32, '-hi': 5, '-wi': 5, '-st':(1, 2), '-po': (3, 4), '-pd': 'VALID'}, # expected failure. see issue https://github.com/xmos/ai_tools/issues/83
        {'-in': 32, '-hi': 4, '-wi': 4, '-st':(2, 2), '-po': (4, 4), '-pd': 'VALID'},
        {'-in': 4, '-hi': 16, '-wi': 16, '-st':(1, 1), '-po': (4, 4), '-pd': 'VALID'},
        {'-in': 32, '-hi': 4, '-wi': 4, '-st':(2, 2), '-po': (2, 2), '-pd': 'VALID'},
        {'-in': 4, '-hi': 16, '-wi': 16, '-st':(1, 1), '-po': (2, 2), '-pd': 'VALID'}
    ]

    if operator in tests or len(tests) == 0:
        test_cases.extend(create_test_cases(operator, generator, parameter_sets))


    #***********************************
    # AvgPool Global
    #***********************************
    operator = operator_codes.XCOREOpCodes.XC_avgpool2d_global.name
    generator = os.path.join(directories.GENERATOR_DIR, 'generate_avgpool2d_global.py')
    parameter_sets = [
        {'-in': 32, '-hi': 4, '-wi': 4},
    ]

    if operator in tests or len(tests) == 0:
        test_cases.extend(create_test_cases(operator, generator, parameter_sets))


    #***********************************
    # Conv2D 1x1
    #***********************************
    operator = operator_codes.XCOREOpCodes.XC_conv2d_1x1.name
    generator = os.path.join(directories.GENERATOR_DIR, 'generate_conv2d_1x1.py')
    parameter_sets = [
        {'-in': 4, '-out': 4, '-hi': 2, '-wi': 2, '--bias_init': ('const', 0), '--weight_init': ('const', 1), '--input_init': ('const', 1)},
        #{'-in': 4, '-out': 4, '-hi': 2, '-wi': 2},
        # {'-in': 4, '-out': 4, '-hi': 5, '-wi': 5},
        # {'-in': 12, '-out': 8, '-hi': 5, '-wi': 5},
        # {'-in': 4, '-out': 4, '-hi': 5, '-wi': 5,},
        # {'-in': 16, '-out': 8, '-hi': 5, '-wi': 5}
    ]

    if operator in tests or len(tests) == 0:
        test_cases.extend(create_test_cases(operator, generator, parameter_sets))


    #***********************************
    # Conv2D deepin/deepout
    #***********************************
    operator = operator_codes.XCOREOpCodes.XC_conv2d_deepin_deepout_relu.name
    generator = os.path.join(directories.GENERATOR_DIR, 'generate_conv2d_deepin_deepout_relu.py')
    parameter_sets = [
        {'-hi': 1, '-wi': 1, '-kh':3, '-kw': 3, '-pd': 'SAME' }, # expected failure. see issue https://github.com/xmos/ai_tools/issues/88
        {'-hi': 3, '-wi': 3, '-kh':3, '-kw': 3, '-pd': 'SAME' },
        {'-hi': 5, '-wi': 5, '-kh':3, '-kw': 3, '-pd': 'SAME' },
        {'-hi': 3, '-wi': 3, '-kh':3, '-kw': 3, '-pd': 'VALID'},
        {'-hi': 5, '-wi': 5, '-kh':3, '-kw': 3, '-pd': 'VALID'},
        {'-hi': 5, '-wi': 5, '-kh':3, '-kw': 3, '-pd': 'VALID', '-par': 2 },
        {'-hi': 5, '-wi': 5, '-kh':3, '-kw': 3, '-pd': 'VALID', '-par': 4 },
        {'-hi': 5, '-wi': 5, '-kh':3, '-kw': 3, '-pd': 'VALID', '-par': 5 }
    ]

    if operator in tests or len(tests) == 0:
        test_cases.extend(create_test_cases(operator, generator, parameter_sets))

    #***********************************
    # Conv2D shallowin/deepout
    #***********************************
    operator = operator_codes.XCOREOpCodes.XC_conv2d_shallowin_deepout_relu.name
    generator = os.path.join(directories.GENERATOR_DIR, 'generate_conv2d_shallowin_deepout_relu.py')
    parameter_sets = [
        {'-hi': 1, '-wi': 1, '-kh':1, '-kw': 1, '-pd': 'SAME' },
        {'-hi': 1, '-wi': 1, '-kh':3, '-kw': 3, '-pd': 'SAME' }, # # expected failure. see issue https://github.com/xmos/ai_tools/issues/88
        {'-hi': 3, '-wi': 3, '-kh':3, '-kw': 3, '-pd': 'SAME' },
        {'-hi': 5, '-wi': 5, '-kh':3, '-kw': 3, '-pd': 'SAME' },
        {'-hi': 1, '-wi': 1, '-kh':1, '-kw': 1, '-pd': 'VALID'},
        {'-hi': 3, '-wi': 3, '-kh':3, '-kw': 3, '-pd': 'VALID'},
        {'-hi': 5, '-wi': 5, '-kh':3, '-kw': 3, '-pd': 'VALID'}
    ]

    if operator in tests or len(tests) == 0:
        test_cases.extend(create_test_cases(operator, generator, parameter_sets))

    #***********************************
    # Fully-connected deepin anyout
    #***********************************
    operator = operator_codes.XCOREOpCodes.XC_fc_deepin_anyout.name
    generator = os.path.join(directories.GENERATOR_DIR, 'generate_fully_connected.py')
    parameter_sets = [
        {'-in': 32, '-out': 16},
        {'-in': 4, '-out': 4},
        {'-in': 4, '-out': 8},
        {'-in': 8, '-out': 4}
    ]
    
    if operator in tests or len(tests) == 0:
        test_cases.extend(create_test_cases(operator, generator, parameter_sets, train_model=True))


    #***********************************
    # Fully-connected deepin anyout requantized
    #***********************************
    operator = operator_codes.XCOREOpCodes.XC_requantize_16_to_8.name
    generator = os.path.join(directories.GENERATOR_DIR, 'generate_fully_connected_requantized.py')
    parameter_sets = [
        {'-in': 32 }
    ]

    if operator in tests or len(tests) == 0:
        test_cases.extend(create_test_cases(operator, generator, parameter_sets, train_model=True))

    #***********************************
    # ArgMax
    #***********************************
    operator = operator_codes.XCOREOpCodes.XC_argmax_16.name
    generator = os.path.join(directories.GENERATOR_DIR, 'generate_argmax_16.py')
    parameter_sets = [
        {'-in': 1 },
        {'-in': 10 },
        {'-in': 100 }
    ]

    if operator in tests or len(tests) == 0:
        test_cases.extend(create_test_cases(operator, generator, parameter_sets))

    #***********************************
    # MaxPool
    #***********************************
    operator = operator_codes.XCOREOpCodes.XC_maxpool2d.name
    generator = os.path.join(directories.GENERATOR_DIR, 'generate_maxpool2d.py')
    parameter_sets = [
        {'-in': 32, '-hi': 2, '-wi': 2, '-st':2, '-po': 2, '-pd': 'VALID'},
        {'-in': 32, '-hi': 4, '-wi': 4, '-st':2, '-po': 2, '-pd': 'VALID'},
        {'-in': 32, '-hi': 16, '-wi': 16, '-st':2, '-po': 2, '-pd': 'VALID'}
    ]

    if operator in tests or len(tests) == 0:
        test_cases.extend(create_test_cases(operator, generator, parameter_sets))


    if not args.dry_run:
        # Remove all existing data
        if os.path.exists(directories.DATA_DIR):
            shutil.rmtree(directories.DATA_DIR)

    # now generate all the test cases
    pool = multiprocessing.Pool(processes=jobs)
    func = partial(generate_test_case, args.dry_run)
    pool.map(func, test_cases)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', action='append', default=[], help="Test to run (defaults to all)")
    parser.add_argument('-j', '--jobs', type=int, default=4, help="Allow N jobs at once")
    parser.add_argument('--dry-run', action='store_true', default=False, help='Perform a dry run.')
    args = parser.parse_args()

    run_generate(args.test, args.jobs)
