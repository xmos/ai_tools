#!/usr/bin/env python
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import sys
import os
import re
import subprocess
from collections import deque, Counter
import argparse
import xml.etree.ElementTree as ET

INDENT = ' ' * 2
XSIM_TRACE_FILENAME = 'xsim_trace.out'
XSIM_TRACE_START_COLUMN = 32

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
            # aggregate 
            aggregate_identifiers = set([c.identifier for c in self.children])
            for aggregate_identifier in sorted(aggregate_identifiers):
                aggregate_context = TraceContext(self.tile, self.core, aggregate_identifier, sys.maxsize)
                aggregate_context.exit_cycle_clock = 0
                for child in self.children:
                    if child.identifier == aggregate_identifier:
                        aggregate_context.entry_cycle_clock = min(
                            child.entry_cycle_clock,
                            aggregate_context.entry_cycle_clock,
                        )
                        aggregate_context.exit_cycle_clock = max(
                            child.exit_cycle_clock,
                            aggregate_context.exit_cycle_clock,
                        )
                        aggregate_context.total_instructions += child.total_instructions
                        aggregate_context.instruction_counter.update(child.instruction_counter)
                        aggregate_context.children.extend(child.children)
                
                child_report = aggregate_context.report(clock_rate, base_indent+INDENT*2)
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

def xsim_bench(args):
    xe_file = os.path.abspath(args.xe)
    output_dir = os.path.abspath(args.output)
    trace_functions = KNOWN_OPERATOR_FUNCTIONS + args.functions # NOTE: not a set because we do regex lookups
    trace_instructions = set(KNOWN_INSTRUCTIONS + args.instructions)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_filename = os.path.join(output_dir, 'xsim_bench.log')

    with open(log_filename, 'w') as log:
        # run xsim
        if args.args:
            cmd = 'xsim --trace --enable-fnop-tracing --args {} {} > {}'.format(xe_file, args.args, XSIM_TRACE_FILENAME)
        else:
            cmd = 'xsim --trace --enable-fnop-tracing {} > {}'.format(xe_file, XSIM_TRACE_FILENAME)
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
                tile = line[:7]
                core = line[8]
                index_plus = line.find('+', XSIM_TRACE_START_COLUMN)
                identifier = line[XSIM_TRACE_START_COLUMN:index_plus].strip()

                if any(re.search(regex, identifier) for regex in trace_functions):
                    index_colon = line.find(':', index_plus)
                    index_whitespace = line.find(' ', index_colon+2)
                    instruction = line[index_colon+2:index_whitespace].strip()

                    if instruction == 'entsp' or instruction == 'dualentsp':
                        index_amp = line.find('@', index_whitespace)
                        entry_cycle_clock = int(line[index_amp+1:].strip())

                        new_context = TraceContext(tile, core, identifier, entry_cycle_clock)
                        new_context.instruction_count = 1

                        if context_stack:
                            new_context.is_child = True
                            context_stack[-1].children.append(new_context)
                        
                        context_stack.append(new_context)
                    elif instruction == 'retsp':
                        index_amp = line.find('@', index_whitespace)
                        exit_cycle_clock = int(line[index_amp+1:].strip())

                        context = context_stack.pop()
                        context.instruction_count += 1
                        context.exit_cycle_clock = exit_cycle_clock
                        
                        if not context.is_child:
                            print(context.report(config['clock_rate']))
                    else:
                        if context_stack:
                            context_stack[-1].total_instructions += 1
                            if instruction in trace_instructions:
                                context_stack[-1].instruction_counter[instruction] += 1
                elif line[12:16] == 'FNOP':
                    if context_stack:
                        if 'FNOP' in trace_instructions:
                            context_stack[-1].instruction_counter['FNOP'] += 1

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
