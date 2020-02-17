#!/usr/bin/env python
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import sys
import os
import re
import subprocess
from collections import deque, Counter
import itertools
import argparse
import xml.etree.ElementTree as ET

INDENT = ' ' * 2
XSIM_TRACE_FILENAME = 'xsim_trace.out'

KNOWN_OPERATOR_FUNCTIONS = [
    'conv2d_deepin_deepout_init',
    'conv2d_deepin_deepout_block.',
    'conv2d_shallowin_deepout_init',
    'conv2d_shallowin_deepout_block.',
    'xcore.+conv.+Eval2D_SIDO',
    'xcore.+conv.+Prepare2D_SIDO',
    'xcore.+conv.+Eval2D_DIDO',
    'xcore.+conv.+Prepare2D_DIDO',
    'xcore.+fully_connected.+Eval_AOI',
    'xcore.+arg_max.+Eval_16',
    'xcore.+max_pool.+Eval',
    'xcore.+avg_pool.+Eval'
]

KNOWN_INSTRUCTIONS = [
    'vlmaccr',
    'vlmacc'
]

class TraceContext():
    def __init__(self, tile, core, identifier, entry_cycle_clock, is_child=False):
        self.tile = tile
        self.core = core
        self.identifier = identifier
        self.entry_cycle_clock = entry_cycle_clock
        self.exit_cycle_clock = None
        self.is_child = is_child

        self.children = []
        self.instruction_counter = Counter()
        self.total_instructions = 0

    def update(self, other):
        self.entry_cycle_clock = min(
            other.entry_cycle_clock,
            self.entry_cycle_clock,
        )
        self.exit_cycle_clock = max(
            other.exit_cycle_clock,
            self.exit_cycle_clock,
        )
        self.total_instructions += other.total_instructions
        self.instruction_counter.update(other.instruction_counter)
        self.children.extend(other.children)

    def report(self, clock_rate, base_indent=''):
        duration = int((self.exit_cycle_clock - self.entry_cycle_clock) / clock_rate)

        # header line
        lines = []
        lines.append(f'{base_indent}[{duration} us]     {str(self)}')
        
        # instruction counts
        line = f'{base_indent}{INDENT}{self.total_instructions} instructions'
        fields = []
        for instruction, count in self.instruction_counter.items():
            fields.append(f'{count} {instruction}')
        line += f' ({", ".join(fields)})'
        lines.append(line)

        # children
        if self.children:
            lines.append(f'{base_indent}{INDENT}children')
            # group children by identifiers
            for _, group in itertools.groupby(self.children, key=lambda c: c.identifier):
                first_child = list(itertools.islice(group, 0, 1))[0]
                # now update the first child with the remainder of the group
                (first_child.update(context) for context in itertools.islice(group, 1))
                
                child_report = first_child.report(clock_rate, base_indent+INDENT*2)
                for line in child_report.split('\n'):
                    lines.append(f'{base_indent}{line}')

        return '\n'.join(lines)

    def __str__(self):
        return f'{self.tile}@{self.core}     {self.identifier}'

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

def parse_trace(data):
    TRACE_START_COLUMN = 32
    tile = data[:7]
    core = data[8]
    index_plus = data.find('+', TRACE_START_COLUMN)
    identifier = data[TRACE_START_COLUMN:index_plus].strip()

    index_colon = data.find(':', index_plus)
    index_whitespace = data.find(' ', index_colon+2)
    instruction = data[index_colon+2:index_whitespace].strip()

    index_amp = data.find('@', index_whitespace)
    cycle_clock = int(data[index_amp+1:].strip())

    return tile, core, identifier, instruction, cycle_clock

def trace_is_fnop(data):
    return data[12:16] == 'FNOP'

def xsim_bench(args):
    xe_file = os.path.abspath(args.xe)
    output_dir = os.path.abspath(args.output)
    trace_functions = [re.compile(tf) for tf in KNOWN_OPERATOR_FUNCTIONS + args.functions]
    trace_instructions = set(KNOWN_INSTRUCTIONS + args.instructions)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_filename = os.path.join(output_dir, 'xsim_bench.log')

    with open(log_filename, 'w') as log:
        # run xsim
        if args.args:
            cmd = 'xsim --trace --enable-fnop-tracing --args {} {} --trace-to {}'.format(xe_file, args.args, XSIM_TRACE_FILENAME)
        else:
            cmd = 'xsim --trace --enable-fnop-tracing {} --trace-to {}'.format(xe_file, XSIM_TRACE_FILENAME)
        print('running: {}'.format(cmd), file=log)
        #FIXME: put these 2 lines back in
        # xsim_output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True, cwd=output_dir)
        # print(xsim_output.decode('utf-8'), file=log)

        # run xobjdump
        cmd = 'xobjdump --split {}'.format(xe_file)
        print('running: {}'.format(cmd), file=log)
        xobjdump_output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True, cwd=output_dir)
        print(xobjdump_output.decode('utf-8'), file=log)

        config = load_config(os.path.join(output_dir, 'config.xml'))

        context_stack = deque()

        # process xsim trace output
        with open(os.path.join(output_dir, XSIM_TRACE_FILENAME)) as fp:
            line = fp.readline()
            while line:
                if trace_is_fnop(line):
                    if context_stack:
                        if 'FNOP' in trace_instructions:
                            context_stack[-1].instruction_counter['FNOP'] += 1
                else:
                    tile, core, identifier, instruction, cycle_clock = parse_trace(line)

                    if any(trace_function.search(identifier) for trace_function in trace_functions):
                        if instruction == 'entsp' or instruction == 'dualentsp':

                            new_context = TraceContext(tile, core, identifier, cycle_clock)
                            new_context.instruction_count = 1

                            if context_stack:
                                new_context.is_child = True
                                context_stack[-1].children.append(new_context)
                            
                            context_stack.append(new_context)
                        elif instruction == 'retsp':

                            context = context_stack.pop()
                            context.instruction_count += 1
                            context.exit_cycle_clock = cycle_clock
                            
                            if not context.is_child:
                                print(context.report(config['clock_rate']))
                        else:
                            if context_stack:
                                context_stack[-1].total_instructions += 1
                                if instruction in trace_instructions:
                                    context_stack[-1].instruction_counter[instruction] += 1

                line = fp.readline()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--xe', required=True, help='Input .xe file')
    parser.add_argument('-a', '--args', help='Argument to pass to .xe file')
    parser.add_argument('-o', '--output', default=os.getcwd(), help='Output directory')
    parser.add_argument('-f', '--function', dest='functions', action='append', default=[], 
        help='Additional function to time')
    parser.add_argument('-i', '--instruction', dest='instructions', action='append', default=[], 
        help='Additional instruction to count')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')
    args = parser.parse_args()

    xsim_bench(args)
