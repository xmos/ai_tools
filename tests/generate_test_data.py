#!/usr/bin/env python

# Copyright (c) 2019, XMOS Ltd, All rights reserved

import os
import sys
import shutil
import subprocess

DATA_DIR = 'data'
if os.path.exists(DATA_DIR):
    shutil.rmtree(DATA_DIR)

SINGLE_OP_MODELS_DIR = os.path.join(DATA_DIR, 'single_op_models')

GENERATOR_DIR = '../python/models_dev/interface_development'


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
        output_dir = os.path.join(SINGLE_OP_MODELS_DIR, operator, folder)
        cmd = f'python {generator} {train_model_flag} {arguments} {output_dir}'
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

#***********************************
# Conv2D deepin/deepout
#***********************************
operator = 'conv2d_deepin_deepout'
generator = os.path.join(GENERATOR_DIR, 'generate_conv2d_deepin_deepout_relu.py')
test_cases = [
    {'hi': 1, 'wi': 1, 'kh':1, 'kw': 1, 'pd': 'SAME' },
    {'hi': 1, 'wi': 1, 'kh':3, 'kw': 3, 'pd': 'SAME' },
    {'hi': 3, 'wi': 3, 'kh':3, 'kw': 3, 'pd': 'SAME' },
    {'hi': 5, 'wi': 5, 'kh':3, 'kw': 3, 'pd': 'SAME' },
    {'hi': 1, 'wi': 1, 'kh':1, 'kw': 1, 'pd': 'VALID' },
    {'hi': 3, 'wi': 3, 'kh':3, 'kw': 3, 'pd': 'VALID' },
    {'hi': 5, 'wi': 5, 'kh':3, 'kw': 3, 'pd': 'VALID' }
]

#generate_test_cases(operator, generator, test_cases)

#***********************************
# Fully-connected final
#***********************************
operator = 'fc_deepin_anyout_final'
generator = os.path.join(GENERATOR_DIR, 'generate_fc_deepin_shallowout.py')
test_cases = [
    {'in': 32 }
]

#generate_test_cases(operator, generator, test_cases, train_model=True)

#***********************************
# ArgMax
#***********************************
operator = 'argmax_16'
generator = os.path.join(GENERATOR_DIR, 'generate_argmax_16.py')
test_cases = [
    {'in': 1 },
    {'in': 10 },
    {'in': 100 }
]

#generate_test_cases(operator, generator, test_cases)

#***********************************
# AvgPool
#***********************************
operator = 'avgpool2d_deep'
generator = os.path.join(GENERATOR_DIR, 'generate_avgpool2d_deep.py')
test_cases = [
    {'in': 32, 'hi': 1, 'wi': 1, 'st':2, 'po': 2, 'pd': 'SAME' },
]

generate_test_cases(operator, generator, test_cases)
