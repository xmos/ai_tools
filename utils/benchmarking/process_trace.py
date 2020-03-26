#!/usr/bin/env python
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import sys
import os
import re
from collections import deque, Counter
import itertools
import argparse

INDENT = ' ' * 2

INSTRUCTIONS = [
    'vlmaccr',
    'vlmacc'
]

class TraceContext():
    def __init__(self, tile, core, identifier, entry_cycle_clock, parent=None, identifier_alt=None):
        self.tile = tile
        self.core = core
        self.identifier = identifier
        self.identifier_alt = identifier_alt
        self.entry_cycle_clock = entry_cycle_clock
        self.exit_cycle_clock = None
        self.parent = parent
        self.par_cores = set([core])

        self.children = []
        self.instruction_counter = Counter()
        self.total_instructions = 0

    @property
    def is_child(self):
        return self.parent is not None

    def update(self, other):
        self.entry_cycle_clock = min(
            other.entry_cycle_clock,
            self.entry_cycle_clock,
        )
        self.exit_cycle_clock = max(
            other.exit_cycle_clock,
            self.exit_cycle_clock,
        )
        self.par_cores.update(other.par_cores)
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
                # now update the first child with the remainder in the group
                for context in itertools.islice(group, 1, None):
                    first_child.update(context)
                
                child_report = first_child.report(clock_rate, base_indent+INDENT*2)
                for line in child_report.split('\n'):
                    lines.append(f'{base_indent}{line}')

        return '\n'.join(lines)

    def __str__(self):
        cores_str = ','.join(sorted(list(self.par_cores)))
        identifier_str = self.identifier_alt or self.identifier
        return f'Tile={self.tile}     Cores={cores_str}     {identifier_str}'

def parse_trace(data):
    TRACE_START_COLUMN = 30
    tile = data[5]
    core = data[8]
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

def process_trace(args):
    # trace_functions = [re.compile(tf) for tf in args.functions]
    trace_functions = []
    trace_functions_alt = {}
    for tf in args.functions:
        if isinstance(tf, tuple):
            tf_c = re.compile(tf[0])
            trace_functions.append(tf_c)
            trace_functions_alt[tf_c] = tf[1]
        else:
            trace_functions.append(re.compile(tf))
    trace_instructions = set(INSTRUCTIONS + args.instructions)

    context_stack = deque()

    # process xsim trace output
    with open(args.trace, 'r') as fd:
        line = fd.readline()
        while line:
            if trace_is_fnop(line):
                if context_stack:
                    if 'FNOP' in trace_instructions:
                        context_stack[-1].instruction_counter['FNOP'] += 1
            else:
                tile, core, identifier, instruction, cycle_clock = parse_trace(line)

                for trace_function in trace_functions:
                    if trace_function.search(identifier):
                        if instruction == 'entsp' or instruction == 'dualentsp':
                            new_context = TraceContext(tile, core, identifier, cycle_clock,
                                identifier_alt=trace_functions_alt.get(trace_function, None))
                            new_context.instruction_count = 1

                            if context_stack:
                                curr_context = context_stack[-1]
                                if new_context.identifier == curr_context.identifier:
                                    new_context.parent = curr_context.parent
                                    curr_context.parent.children.append(new_context)
                                else:
                                    new_context.parent = curr_context
                                    curr_context.children.append(new_context)

                            context_stack.append(new_context)
                        elif instruction == 'retsp':
                            context = context_stack.pop()
                            context.instruction_count += 1
                            context.exit_cycle_clock = cycle_clock
                            
                            if not context.is_child:
                                print(context.report(args.clock_rate))
                                print()
                        else:
                            if context_stack:
                                context_stack[-1].total_instructions += 1
                                if instruction in trace_instructions:
                                    context_stack[-1].instruction_counter[instruction] += 1

            line = fd.readline()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trace', required=True, help='Input trace file file')
    parser.add_argument('-c', '--clock-rate', dest='clock_rate', type=int, 
        default=800, help='Clock rate (default is 800 MHz)')
    parser.add_argument('-f', '--function', dest='functions', action='append', default=[], 
        help='Additional function to time')
    parser.add_argument('--trace-functions', dest='trace_functions', default='trace_functions.txt',
        help='File of additional functions to time')
    parser.add_argument('-i', '--instruction', dest='instructions', action='append', default=[], 
        help='Additional instruction to count')
    args = parser.parse_args()

    # load trace functions
    with open(args.trace_functions, 'r') as fd:
        line = fd.readline()
        while line:
            if not line.startswith('#'):
                fields = line.strip().split(',')
                if len(fields) == 1:
                    args.functions.append(fields[0].strip())
                elif len(fields) > 1:
                    args.functions.append((fields[0].strip(), fields[1].strip()))
            line = fd.readline()

    process_trace(args)
