#!/usr/bin/env python
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import sys
import os
import re
from collections import deque, Counter
import itertools
import argparse

INDENT = ' ' * 2

# INSTRUCTIONS = [
#     'vlmaccr',
#     'vlmacc'
# ]

class TraceContext():
    # def __init__(self, tile, core, identifier, entry_cycle_clock, parent=None, identifier_alt=None):
    def __init__(self, tile, core, identifier, entry_cycle_clock, identifier_alt=None, thread_worker=None):
        self.tile = tile
        self.core = core
        self.identifier = identifier
        self.identifier_alt = identifier_alt
        self.thread_worker = thread_worker
        self.entry_cycle_clock = entry_cycle_clock
        self.exit_cycle_clock = None
        self.thread_cores = set([core])

    def report(self, clock_rate, base_indent=''):
        duration = int((self.exit_cycle_clock - self.entry_cycle_clock) / clock_rate)

        return f'{base_indent}[{duration} us]     {str(self)}'

    def __str__(self):
        cores_str = ','.join([str(c) for c in sorted(list(self.thread_cores))])
        identifier_str = self.identifier_alt or self.identifier
        return f'Tile={self.tile}     Cores={cores_str}     {identifier_str}'

def parse_trace(data):
    TRACE_START_COLUMN = 30
    tile = int(data[5])
    core = int(data[8])
    index_open_paren = data.find('(', TRACE_START_COLUMN)
    index_plus = data.find('+', index_open_paren+1)
    identifier = data[index_open_paren+1:index_plus].strip()

    index_colon = data.find(':', index_plus)
    index_whitespace = data.find(' ', index_colon+2)
    instruction = data[index_colon+2:index_whitespace].strip()

    index_amp = data.find('@', index_whitespace)
    cycle_clock = int(data[index_amp+1:].strip())

    return tile, core, identifier, instruction, cycle_clock

def trace_is_fnop(data):
    return data[12:16] == 'FNOP'

def process_trace(args,functions):
    trace_functions_re = []
    thread_worker_functions_re = []
    thread_worker_functions_lut = {}

    trace_functions_alt = {}
    for tf in functions:
        tf_c = re.compile(tf[0])
        trace_functions_re.append(tf_c)
        if tf[2]:
            wf_c = re.compile(tf[2])
            thread_worker_functions_re.append(wf_c)
            thread_worker_functions_lut[tf_c] = wf_c

        trace_functions_alt[tf_c] = tf[1]

    context_stack = [deque()] * 8

    # process xsim trace output
    with open(args.trace, 'r') as fd:
        line = fd.readline()
        while line:
            if not trace_is_fnop(line):
                tile, core, identifier, instruction, cycle_clock = parse_trace(line)

                for trace_function in trace_functions_re:
                    if trace_function.search(identifier):
                        if instruction == 'entsp' or instruction == 'dualentsp':
                            new_context = TraceContext(tile, core, identifier, cycle_clock,
                                identifier_alt=trace_functions_alt.get(trace_function, None),
                                thread_worker=thread_worker_functions_lut.get(trace_function, None))
                            context_stack[core].append(new_context)
                        elif instruction == 'retsp':
                            context = context_stack[core].pop()
                            context.exit_cycle_clock = cycle_clock
                            print(context.report(args.clock_rate))
                for thread_worker in thread_worker_functions_re:
                    if thread_worker.search(identifier):
                        if instruction == 'entsp' or instruction == 'dualentsp':
                            curr_context = context_stack[core][-1]
                            if thread_worker == curr_context.thread_worker:
                                curr_context.thread_cores.add(core)

            line = fd.readline()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trace', required=True, help='Input trace file file')
    parser.add_argument('-c', '--clock-rate', dest='clock_rate', type=int, 
        default=800, help='Clock rate (default is 800 MHz)')
    parser.add_argument('--trace-functions', dest='trace_functions', default='trace_functions.txt',
        help='File of additional functions to time')
    args = parser.parse_args()

    # load trace functions
    functions = []
    with open(args.trace_functions, 'r') as fd:
        line = fd.readline()
        while line:
            if not line.startswith('#'):
                fields = line.strip().split(',')
                if len(fields) == 1:
                    functions.append((fields[0].strip(), None, None))
                elif len(fields) == 2:
                    functions.append((fields[0].strip(), fields[1].strip(), None))
                elif len(fields) > 2:
                    functions.append((fields[0].strip(), fields[1].strip(), fields[2].strip()))
            line = fd.readline()

    process_trace(args, functions)
