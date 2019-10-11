#!/usr/bin/env python
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import sys
import os
import subprocess
import argparse
import xml.etree.ElementTree as ET

def is_operator(record):
    known_operators = [
        'fc_deepin_shallowout_lin',
        'conv2d_deepin_deepout_relu',
        'conv2d_shallowin_deepout_relu',
        'maxpool2d_deep',
        'tflite::reference_ops::FullyConnected'
    ]

    for operator in known_operators:
        #print(record, operator)
        if record.startswith(operator):
            return True

    return False

def iter_config(config_filename):
    tree = ET.parse(config_filename)
    root = tree.getroot()

    for node in root.findall('System/Nodes/Node'):
        for proc in node.findall('Processor'):
            tile = proc.get('codeReference')
            cores = int(proc.get('numThreads'))
            for core in range(cores):
                yield tile, str(core)

def xsim_bench(args):
    # xe_file = os.path.abspath(args.xe)
    # output_dir = os.path.abspath(args.output)

    xe_file = args.xe
    output_dir = os.path.abspath(args.output)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_filename = os.path.join(output_dir, 'xsim_bench.log')

    with open(log_filename, 'w') as log:
        # run xsim
        if args.args:
            cmd = 'xsim --gprof --args {} {}'.format(xe_file, args.args)
        else:
            cmd = 'xsim --gprof {}'.format(xe_file)
        print('running: {}'.format(cmd), file=log)
        xsim_output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True, cwd=output_dir)
        print(xsim_output.decode('utf-8'), file=log)

        # run xobjdump
        cmd = 'xobjdump --split {}'.format(xe_file)
        print('running: {}'.format(cmd), file=log)
        xobjdump_output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True, cwd=output_dir)
        print(xobjdump_output.decode('utf-8'), file=log)

        # run xgprof
        config_filename = os.path.join(output_dir, 'config.xml')

        for tile, core in iter_config(config_filename):
            gprof_filename = '{}_core{}.gprof'.format(tile, core)
            if os.path.exists(os.path.join(output_dir, gprof_filename)):
                cmd = 'xgprof --flat-profile --brief --demangle image_n0c0.elf {}'.format(gprof_filename)
                print('running: {}'.format(cmd), file=log)
                xgprof_output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True, cwd=output_dir)
                print(xgprof_output.decode('utf-8'), file=log)

                xgprof_lines = xgprof_output.decode('utf-8').split('\n')
                for i_line, line in enumerate(xgprof_lines):
                    if line.startswith('Flat profile:'):
                        break

                # determine duration units
                fields = xgprof_lines[i_line+4].split()
                if fields[4] == 'mm/call' or fields[4] == 'um/call':
                    units = 'Microseconds'
                elif fields[4] == 'ms/call':
                    units = 'Milliseconds'
                else:
                    print(f'Unsupported units {fields[4]}', file=sys.stderr)

                template = '{:40}{:10}{:10}{:10}'
                print_header = True
                for xgprof_line in xgprof_lines[i_line+5:]:
                    fields = xgprof_line.split()
                    if fields:
                        if len(fields) >= 7:
                            name = fields[6].strip()
                            if is_operator(name):
                                if units == 'Microseconds':
                                    duration = float(fields[4]) / 1000.0
                                else: # must be Milliseconds
                                    duration = float(fields[4])
                                if print_header:
                                    print(template.format('Operator', 'Tile', 'Core', 'Milliseconds'))
                                    print_header = False
                                print(template.format(name, tile, core, duration))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--xe', required=True, help='Input .xe file')
    parser.add_argument('-a', '--args', help='Argument to pass to .xe file')
    parser.add_argument('-o', '--output', default=os.getcwd(), help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')
    args = parser.parse_args()

    xsim_bench(args)
