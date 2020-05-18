#!/usr/bin/env python
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import argparse
import re
from trace_context import TraceContextTree
from trace_parser import XCoreOperation


def process_trace(fnames, trace_file):
    context_tree = None
    
    cycles = []

    with open(trace_file, 'r') as fd:

        for op in XCoreOperation.read_all(fd):
            
            if (context_tree is None):
                if not op.is_frame_entry():
                    continue
                if op.parent_symbol in fnames:
                    context_tree = TraceContextTree(None, op)
            else:
                new_tree = context_tree.process_op(op)

                if (new_tree is None):
                    t = context_tree.op_count()
                    cycles.append(t)
                    # t = context_tree.exit_clock - context_tree.entry_clock
                    # cycles.append(t//5)
                context_tree = new_tree

    return cycles
