#!/usr/bin/env python
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import sys
import os
import re
from collections import deque, Counter
import itertools
import argparse

INDENT = ' ' * 2

KNOWN_OPERATOR_FUNCTIONS = [
    'argmax_16',
    'conv2d_deepin_deepout_init',
    'conv2d_deepin_deepout_block.',
    'conv2d_shallowin_deepout_init',
    'conv2d_shallowin_deepout_block.',
    'fully_connected_init',
    'fully_connected_16',
    'avgpool2d',
    'avgpool2d_global',
    'maxpool2d',
    'requantize_16_to_8',
    'xcore.+argmax.+Eval_ArgMax_16',
    'xcore.+conv.+Eval2D_SIDO',
    'xcore.+conv.+Prepare2D_SIDO',
    'xcore.+conv.+Eval2D_DIDO',
    'xcore.+conv.+Prepare2D_DIDO',
    'xcore.+fully_connected.+Eval_16',
    'xcore.+maxpool.+Eval2D',
    'xcore.+avgpool_global.+Eval2D',
    'xcore.+type_conversions.+Eval_Requantize_16_to_8',
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

def process_trace(args):
    trace_functions = [re.compile(tf) for tf in KNOWN_OPERATOR_FUNCTIONS + args.functions]
    trace_instructions = set(KNOWN_INSTRUCTIONS + args.instructions)

    context_stack = deque()

    # process xsim trace output
    with open(args.trace, 'r') as fp:
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
                            print(context.report(args.clock_rate))
                    else:
                        if context_stack:
                            context_stack[-1].total_instructions += 1
                            if instruction in trace_instructions:
                                context_stack[-1].instruction_counter[instruction] += 1

            line = fp.readline()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trace', required=True, help='Input trace file file')
    parser.add_argument('-c', '--clock-rate', dest='clock_rate', type=int, 
        default=800, help='Clock rate (default is 800 MHz)')
    parser.add_argument('-f', '--function', dest='functions', action='append', default=[], 
        help='Additional function to time')
    parser.add_argument('-i', '--instruction', dest='instructions', action='append', default=[], 
        help='Additional instruction to count')
    args = parser.parse_args()

    process_trace(args)
