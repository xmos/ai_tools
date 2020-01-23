#!/usr/bin/env python

# Copyright (c) 2019, XMOS Ltd, All rights reserved

import os
import sys
import shutil
import subprocess

import directories
from operator_codes import XCOREOpCodes

def make_folder_and_arguments(**kwargs):
    folder_fields = []
    aurgment_fields = []
    for key, value in kwargs.items():
        folder_fields.append(f'{key}_{value}')
        aurgment_fields.append(f'-{key} {value}')

    return '-'.join(folder_fields), ' '.join(aurgment_fields)

def generate_test_cases(operator, generator, test_cases, *, train_model=False):
    if train_model:
        train_model_flag = '--train_model'
    else:
        train_model_flag = ''

    for test_case in test_cases:
        folder, arguments = make_folder_and_arguments(**test_case)
        output_dir = os.path.join(directories.SINGLE_OP_MODELS_DATA_DIR, operator, folder)
        cmd = f'python {generator} {train_model_flag} {arguments} {output_dir}'
        print(f'generating test case {output_dir}')
        try:
            subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as cmdexc:                                                                                                   
            print(cmdexc.output.decode('utf-8'))

#***********************************
# Remove all existing data
#***********************************
if os.path.exists(directories.DATA_DIR):
    shutil.rmtree(directories.DATA_DIR)

#***********************************
# Conv2D deepin/deepout
#***********************************
operator = XCOREOpCodes.XC_conv2d_deepin_deepout_relu.name
generator = os.path.join(directories.GENERATOR_DIR, 'generate_conv2d_deepin_deepout_relu.py')
test_cases = [
    {'hi': 1, 'wi': 1, 'kh':1, 'kw': 1, 'pd': 'SAME' },
    # {'hi': 1, 'wi': 1, 'kh':3, 'kw': 3, 'pd': 'SAME' },
    {'hi': 3, 'wi': 3, 'kh':1, 'kw': 1, 'pd': 'SAME' },
    {'hi': 3, 'wi': 3, 'kh':3, 'kw': 3, 'pd': 'SAME' },
    {'hi': 5, 'wi': 5, 'kh':3, 'kw': 3, 'pd': 'SAME' },
    {'hi': 1, 'wi': 1, 'kh':1, 'kw': 1, 'pd': 'VALID' },
    {'hi': 3, 'wi': 3, 'kh':3, 'kw': 3, 'pd': 'VALID' },
    {'hi': 5, 'wi': 5, 'kh':3, 'kw': 3, 'pd': 'VALID' }
]

generate_test_cases(operator, generator, test_cases)

#***********************************
# Conv2D shallowin/deepout
#***********************************
operator = XCOREOpCodes.XC_conv2d_shallowin_deepout_relu.name
generator = os.path.join(directories.GENERATOR_DIR, 'generate_conv2d_shallowin_deepout_relu.py')
test_cases = [
    {'hi': 1, 'wi': 1, 'kh':1, 'kw': 1, 'pd': 'SAME' },
    # {'hi': 1, 'wi': 1, 'kh':3, 'kw': 3, 'pd': 'SAME' },
    {'hi': 3, 'wi': 3, 'kh':3, 'kw': 3, 'pd': 'SAME' },
    {'hi': 5, 'wi': 5, 'kh':3, 'kw': 3, 'pd': 'SAME' },
    {'hi': 1, 'wi': 1, 'kh':1, 'kw': 1, 'pd': 'VALID' },
    {'hi': 3, 'wi': 3, 'kh':3, 'kw': 3, 'pd': 'VALID' },
    {'hi': 5, 'wi': 5, 'kh':3, 'kw': 3, 'pd': 'VALID' }
]

generate_test_cases(operator, generator, test_cases)

#***********************************
# Fully-connected final
#***********************************
operator = XCOREOpCodes.XC_fc_deepin_anyout_final.name
generator = os.path.join(directories.GENERATOR_DIR, 'generate_fc_deepin_shallowout.py')
test_cases = [
    {'in': 32 }
]

generate_test_cases(operator, generator, test_cases, train_model=True)

#***********************************
# ArgMax
#***********************************
operator = XCOREOpCodes.XC_argmax_16.name
generator = os.path.join(directories.GENERATOR_DIR, 'generate_argmax_16.py')
test_cases = [
    {'in': 1 },
    {'in': 10 },
    {'in': 100 }
]

generate_test_cases(operator, generator, test_cases)

#***********************************
# AvgPool
#***********************************
operator = XCOREOpCodes.XC_avgpool2d_deep.name
generator = os.path.join(directories.GENERATOR_DIR, 'generate_avgpool2d_deep.py')
test_cases = [
    {'in': 32, 'hi': 2, 'wi': 2, 'st':2, 'po': 2, 'pd': 'VALID' },
    {'in': 32, 'hi': 4, 'wi': 4, 'st':2, 'po': 2, 'pd': 'VALID' },
    {'in': 32, 'hi': 16, 'wi': 16, 'st':2, 'po': 2, 'pd': 'VALID' }
]

generate_test_cases(operator, generator, test_cases)

#***********************************
# MaxPool
#***********************************
operator = XCOREOpCodes.XC_maxpool2d_deep.name
generator = os.path.join(directories.GENERATOR_DIR, 'generate_maxpool2d_deep.py')
test_cases = [
    {'in': 32, 'hi': 2, 'wi': 2, 'st':2, 'po': 2, 'pd': 'VALID' },
    {'in': 32, 'hi': 4, 'wi': 4, 'st':2, 'po': 2, 'pd': 'VALID' },
    {'in': 32, 'hi': 16, 'wi': 16, 'st':2, 'po': 2, 'pd': 'VALID' }
]

generate_test_cases(operator, generator, test_cases)
