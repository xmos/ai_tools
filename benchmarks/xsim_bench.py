#!/usr/bin/env python
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import sys
import os
import subprocess
import argparse
import xml.etree.ElementTree as ET

XSIM_TRACE_FILENAME = 'xsim_trace.out'
XSIM_TRACE_START_COLUMN = 32

KNOWN_OPERATOR_FUNCTIONS = [
    'fc_deepin_shallowout_lin_asm',
    'fc_deepin_shallowout_lin_c',
    'conv2d_deepin_deepout_relu_asm',
    'conv2d_deepin_deepout_relu_c',
    'conv2d_shallowin_deepout_relu_asm',
    'conv2d_shallowin_deepout_relu_c',
    'maxpool2d_deep_asm',
    'maxpool2d_deep_c'
]

def load_config(config_filename):
    config = {
        'clock_rate': None,
        'tiles': []
    }
    tree = ET.parse(config_filename)
    root = tree.getroot()

    for node in root.findall('System/Nodes/Node'):
        config['clock_rate'] = int(node.get('processorMhz'))
        for proc in node.findall('Processor'):
            config['tiles'].append({
                'tile': proc.get('codeReference'),
                'cores': int(proc.get('numThreads'))
            })
    
    return config

def xsim_bench(args):
    xe_file = os.path.abspath(args.xe)
    output_dir = os.path.abspath(args.output)
    trace_functions = KNOWN_OPERATOR_FUNCTIONS + args.functions

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_filename = os.path.join(output_dir, 'xsim_bench.log')

    with open(log_filename, 'w') as log:
        # run xsim
        if args.args:
            cmd = 'xsim -t --args {} {} > {}'.format(xe_file, args.args, XSIM_TRACE_FILENAME)
        else:
            cmd = 'xsim -t {} > {}'.format(xe_file, XSIM_TRACE_FILENAME)
        print('running: {}'.format(cmd), file=log)
        xsim_output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True, cwd=output_dir)
        print(xsim_output.decode('utf-8'), file=log)

        # run xobjdump
        cmd = 'xobjdump --split {}'.format(xe_file)
        print('running: {}'.format(cmd), file=log)
        xobjdump_output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True, cwd=output_dir)
        print(xobjdump_output.decode('utf-8'), file=log)

        config = load_config(os.path.join(output_dir, 'config.xml'))

        # process xsim trace output
        tracked_operators = {}

        with open(os.path.join(output_dir, XSIM_TRACE_FILENAME)) as fp:
            line = fp.readline()
            while line:
                index_plus = line.find('+', XSIM_TRACE_START_COLUMN)
                trace_function = line[XSIM_TRACE_START_COLUMN:index_plus].strip()
                if trace_function in trace_functions:
                    index_colon = line.find(':', index_plus)
                    index_whitespace = line.find(' ', index_colon+2)

                    instruction = line[index_colon+2:index_whitespace].strip()
                    tile = line[:7]
                    core = line[8]

                    tracking_key = (tile, core, trace_function)

                    if tracking_key not in tracked_operators:
                        # operator function entry
                        index_amp = line.find('@', index_whitespace)
                        entry_cycle_clock = int(line[index_amp+1:].strip())
                        tracked_operator ={
                            'identifier': trace_function,
                            'entry_cycle_clock': entry_cycle_clock,
                            'tiles': {
                                tile: set([core])
                            }
                        }
                        tracked_operators[tracking_key] = tracked_operator
                    elif instruction == 'retsp':
                        # operator function exit
                        tracked_operator = tracked_operators[tracking_key]
                        identifier = tracked_operator['identifier']
                        entry_cycle_clock = tracked_operator['entry_cycle_clock']

                        index_amp = line.find('@', index_whitespace)
                        exit_cycle_clock = int(line[index_amp+1:].strip())
                        duration = int((exit_cycle_clock - entry_cycle_clock) / config['clock_rate'])

                        # print report
                        print(f'{identifier}: {duration} us')
                        for tile_id, cores_used in tracked_operator['tiles'].items():
                            cores_used_str = ','.join(cores_used)
                            print(f'   {tile_id}  cores: {cores_used_str}')

                        # remove tracked operator
                        del tracked_operators[tracking_key]
                    else:
                        tracked_operator = tracked_operators[tracking_key]
                        tracked_operator['tiles'][tile].add(core)

                line = fp.readline()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--xe', required=True, help='Input .xe file')
    parser.add_argument('-a', '--args', help='Argument to pass to .xe file')
    parser.add_argument('-o', '--output', default=os.getcwd(), help='Output directory')
    parser.add_argument('-f', '--function', dest='functions', action='append', default=[], help='Additional function to time')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')
    args = parser.parse_args()

    xsim_bench(args)
