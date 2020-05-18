
from trace_parser import InstructionFetch

class TraceContextTree(object):

    def __init__(self, parent, op):
        assert(op.is_frame_entry())

        self.parent = parent
        self.children = []

        self.histogram = {'fnop': 0}

        self.tile = op.tile
        self.core = op.core
        self.symbol_name = op.parent_symbol
        self.entry_clock = op.clock
        self.exit_clock = None
        self.exclusive_ops = 0

        self.update_hist(op)

    def update_hist(self, op):
        
        self.exclusive_ops += 1

        if isinstance(op, InstructionFetch):
            self.histogram['fnop'] = self.histogram['fnop'] + 1
            return

        for inst in op.instructions():
            if (inst in self.histogram):
                self.histogram[inst] = self.histogram[inst] + 1
            else:
                self.histogram[inst] = 1

    def process_op(self, op):
        if op.is_frame_entry():
            child = TraceContextTree(self, op)
            self.children.append(child)
            return child

        self.update_hist(op)

        if op.is_frame_exit():
            self.exit_clock = op.clock
            return self.parent

        return self

    def op_count(self, include_children = True):
        total = self.exclusive_ops
        if not include_children: return total

        for child in self.children:
            total += child.op_count()
        return total

    def instruction_count(self, *instructions, include_children = True):
        total = 0

        for x in instructions:
            if x not in self.histogram:
                continue

            total += self.histogram[x]

        if include_children:
            for child in self.children:
                total += child.instruction_count(instructions, True)
        
        return total