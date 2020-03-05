#!/usr/bin/env python

# Copyright (c) 2019, XMOS Ltd, All rights reserved

import os
import sys
import re
import json
import shutil
from functools import partial
import subprocess
import multiprocessing
import argparse

import directories

stdout_lock = multiprocessing.Lock()

def make_folder_and_arguments(**kwargs):
    folder_fields = []
    aurgment_fields = []
    compiled_re = re.compile('(?<=-)\w+')

    if kwargs:
        for key, value in kwargs.items():
            hyphenless_key = compiled_re.search(key).group(0) # strip off leading hyphens
            if isinstance(value, list):
                value_folder_str = 'x'.join([str(v) for v in value])
                value_argument_str = ' '.join([str(v) for v in value])
            else:
                value_folder_str = str(value)
                value_argument_str = str(value)

            folder_fields.append(f'{hyphenless_key}={value_folder_str}')
            aurgment_fields.append(f'{key} {value_argument_str}')

        return '_'.join(folder_fields), ' '.join(aurgment_fields)
    else:
        return 'defaults', ''

def generate_test_case(dry_run, test_case):
    operator = test_case['operator']
    generator = test_case['generator']
    if 'parameters' in test_case:
        parameters = test_case['parameters']
    else:
        parameters = {}

    if 'train_model' in test_case and test_case['train_model']:
        train_model_flag = '--train_model'
    else:
        train_model_flag = ''

    folder, arguments = make_folder_and_arguments(**parameters)
    output_dir = os.path.join(directories.OP_TEST_MODELS_DATA_DIR, operator, folder)
    cmd = f'python {generator} {train_model_flag} {arguments} -path {output_dir}'
    with stdout_lock:
        print(f'running: {cmd}')
    if not dry_run:
        try:
            subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as cmdexc:                                                                                                   
            print(cmdexc.output.decode('utf-8'))

def run_generate(test_file, jobs):
    with open(test_file, 'r') as fd:
        test_cases = json.loads(fd.read())

        # Remove all existing data
        if not args.dry_run:
            if os.path.exists(directories.DATA_DIR):
                shutil.rmtree(directories.DATA_DIR)

        # now generate all the test cases
        pool = multiprocessing.Pool(processes=jobs)
        func = partial(generate_test_case, args.dry_run)
        pool.map(func, test_cases)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test-file', default='all_tests.json', help="Test to run (defaults to all_tests.json)")
    parser.add_argument('-n', '--jobs', type=int, default=multiprocessing.cpu_count(), help="Allow N jobs at once")
    parser.add_argument('--smoke', action='store_true', default=False, help='Run smoke tests (smoke_tests.json)')
    parser.add_argument('--dry-run', action='store_true', default=False, help='Perform a dry run')
    args = parser.parse_args()

    if args.smoke:
        args.test_file = 'smoke_tests.json'

    run_generate(args.test_file, args.jobs)
