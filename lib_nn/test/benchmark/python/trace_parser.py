#!/usr/bin/env python
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import re


FRAME_ENTRY_INSTRUCTIONS = [ 'entsp', 'dualentsp' ]
FRAME_EXIT_INSTRUCTIONS = ['retsp']

class InstLineTokens(object):
    re_inst = re.compile(r"^([Aap-]+.*-.----)\.*([0-9a-f]{8}) \(([\S]+)\b\s*\+\s*([0-9a-fA-F]+)\) : (.*) @(\d+)$")

    def __init__(self, line):
        super(InstLineTokens, self).__init__()

        self.tile = int(line[5])
        self.core = int(line[8])
        mode_flag = line[11:14]
        assert(mode_flag in ['-SI','-DI'])
        self.dual_issue = mode_flag == '-DI'
        line = line[15:]

        m = InstLineTokens.re_inst.match(line.strip())
        assert(m is not None)

        self.flags = m.group(1)
        self.pc = int(m.group(2),base=16)
        self.symbol = m.group(3)
        self.symbol_offset = int(m.group(4),base=16)
        self.instruction = m.group(5).split()
        self.clock = int(m.group(6))


class XCoreOperation(object):

    def __init__(self, tile, core):

        super(XCoreOperation, self).__init__()

        self.tile = tile
        self.core = core
        self.clock = None

    @property
    def isFNOP(self):
        return isinstance(self, InstructionFetch)

    @property
    def isDualIssue(self):
        return isinstance(self, DualIssueInstruction)

    @property
    def tile_core(self):
        return "tile[{}].core[{}]".format(self.tile, self.core)

    def contains(self, instruction_pneumonic):
        raise NotImplementedError()

    def is_frame_entry(self):
        raise NotImplementedError()

    def is_frame_exit(self):
        raise NotImplementedError()

    def __str__(self):
        return "@{}:\ttile[{}].core[{}]: FNOP".format(self.clock, self.tile, self.core)


    @staticmethod
    def read_all(file):

        line = file.readline()

        while line:
            if "ECALL" in line:
                # An exception happened in xsim
                raise Exception("trace file ended with an exception ({})".format(line))
            #Would prefer regex, but that could slow parsing down considerably.
            is_fnop = line[9] is not '-'
            
            if is_fnop:
                tile = int(line[5])
                core = int(line[8])
                yield InstructionFetch(tile, core, line)
            else:
                tokens1 = InstLineTokens(line)

                # Check if dual issue
                if not tokens1.dual_issue:
                    yield XCoreSingleInstruction(tokens1)
                else:
                    #need to determine whether it was actually bundled with the 
                    # following instruction
                    line = file.readline()
                    is_fnop = line[9] is not '-'

                    if is_fnop:
                        yield XCoreSingleInstruction(tokens1)
                        continue
                    
                    tokens2 = InstLineTokens(line)
                    if XCoreBundledInstruction.are_bundled(tokens1, tokens2):
                        yield XCoreBundledInstruction(tokens1, tokens2)
                    else:
                        yield XCoreSingleInstruction(tokens1)
                        continue
            line = file.readline()


class InstructionFetch(XCoreOperation):
    def __init__(self, tile, core, line):
        super(InstructionFetch, self).__init__(tile, core)
        self.clock = int(line.split()[-1][1:])

    def contains(self, instruction_pneumonic):
        return False

    def is_frame_entry(self):
        return False

    def is_frame_exit(self):
        return False

class XCoreInstruction(XCoreOperation):
    def __init__(self, tokens):
        super(XCoreInstruction, self).__init__(tokens.tile, tokens.core)
        self.pc = tokens.pc
        self.parent_symbol = tokens.symbol
        self.parent_offset = tokens.symbol_offset
        self.clock = tokens.clock

    def instructions(self):
        raise NotImplementedError()

    def contains(self, instruction_pneumonic):
        for inst in self.instructions():
            if instruction_pneumonic == inst:
                return True
        return False

    def is_frame_entry(self):
        for inst in self.instructions():
            if inst in FRAME_ENTRY_INSTRUCTIONS:
                return True
        return False

    def is_frame_exit(self):
        for inst in self.instructions():
            if inst in FRAME_EXIT_INSTRUCTIONS:
                return True
        return False

class XCoreSingleInstruction(XCoreInstruction):
    def __init__(self, tokens):
        super(XCoreSingleInstruction, self).__init__(tokens)
        self.flags = tokens.flags
        self.instruction = tokens.instruction[0]
        self.args = tokens.instruction[1:]

    def instructions(self):
        yield self.instruction

    def __str__(self):
        return "@{}:\ttile[{}].core[{}]: {}".format(self.clock, self.tile, self.core, self.instruction)


class XCoreBundledInstruction(XCoreInstruction):
    def __init__(self, tokens1, tokens2):
        super(XCoreBundledInstruction, self).__init__(tokens1)

        self.flags = (tokens1.flags, tokens2.flags)
        self.instruction = (tokens1.instruction[0], tokens2.instruction[0])
        self.args = (tokens1.instruction[1:], tokens2.instruction[1:])
        self.clocks = (tokens1.clock, tokens2.clock)

    def instructions(self):
        for inst in self.instruction:
            yield inst

    def __str__(self):
        return "@{}:\ttile[{}].core[{}]: ({}, {})".format(self.clock, self.tile, self.core, self.instruction[0], self.instruction[1])

    @staticmethod
    def are_bundled(token1, token2):
        if not (token1.tile == token2.tile and token1.core == token2.core):
            return False
        if not (token1.dual_issue and token2.dual_issue):
            return False
        # if not ((token1.pc % 4) == 0): #not sure this is required.
        #     return False
        if not ((token2.pc - token1.pc) == 2):
            return False
        if (token2.clock - token1.clock > 5):
            return False
        if (token2.symbol != token1.symbol):
            return False
        
        return True