# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import os
import struct
import re

class CFile():
    def __init__(self, name, includes=None, initializers=None, functions=None, model=None):

        basename, extension = os.path.splitext(name)

        self.source_filename = f'{basename}.c'
        self.header_filename = f'{basename}.h'

        self.includes = includes or []
        self.initializers = initializers or []
        self.functions = functions or []
        self.model = model
        
        self._build_macro_lookup()

    def _build_macro_lookup(self):
        self._macro_lookup = {}
        for initializer in self.initializers:
            name = initializer.GetSanitizedName()
            identifier = name.upper()  #TODO: is this safe, might need to check that result
                                              #      is a valid macro
            replacement_list = None
            if self.model:
                buffer = self.model.GetBuffer(initializer.GetBuffer())
                stdtype = initializer.GetStandardType()
                if stdtype == 'int8_t':
                    values = buffer
                elif stdtype == 'int16_t':
                    values = [i[0] for i in struct.iter_unpack('h', bytearray(buffer))]
                elif stdtype == 'int32_t':
                    values = [i[0] for i in struct.iter_unpack('i', bytearray(buffer))]

                replacement_list = ', '.join([str(v) for v in values])

            self._macro_lookup[name] = {'identifier': identifier, 'replacement-list': replacement_list}

    def add_include(self, include):
        self.includes.append(include)

    def add_initializer(self, initializer):
        self.initializers.append(initializer)
        self._build_macro_lookup()

    def add_function(self, funct):
        self.functions.append(funct)

    def render_header(self):
        lines = []

        include_guard = re.sub('[^0-9a-zA-Z]+', '_', self.header_filename).upper()
        lines.append(f'#ifndef {include_guard}')
        lines.append(f'#define {include_guard}')
        lines.append('')

        for initializer in self.initializers:
            name = initializer.GetSanitizedName()
            macro = self._macro_lookup[name]
            if macro['replacement-list']:
                identifier = macro['identifier']
                replacement_list = macro['replacement-list']
                lines.append(f'#define {identifier} {replacement_list}')
            lines.append('')

        for function in self.functions:
            lines.append(function.render_typedefs())
        lines.append('')

        for function in self.functions:
            lines.append(function.render_declaration())
        lines.append('')

        #endif /* GRANDPARENT_H */
        lines.append(f'#endif /* {include_guard} */')

        return '\n'.join(lines)

    def render_source(self):
        lines = []

        for include in self.includes:
            lines.append(f'#include "{include}"')
        lines.append(f'#include "{self.header_filename}"')
        lines.append('')

        lines.append(f'#ifdef __XC__')
        lines.append(f'#define WORD_ALIGNED [[aligned(4)]]')
        lines.append(f'#else')
        lines.append(f'#define WORD_ALIGNED')
        lines.append(f'#endif')
        lines.append('')

        # initializers
        for initializer in self.initializers:
            name = initializer.GetSanitizedName()
            stdtype = initializer.GetStandardType()
            shape = initializer.GetShape()
            dims = ' * '.join([str(v) for v in shape])

            macro = self._macro_lookup[name]
            if macro['replacement-list']:
                identifier = macro['identifier']
                rhs = f' = {{{identifier}}}'
            else:
                rhs = ''
            lines.append(f'const {stdtype} WORD_ALIGNED {name}[{dims}]{rhs};')
        lines.append('')

        # functions
        for function in self.functions:
            lines.append(function.render_signature())
            lines.append(function.render_body())

        lines.append('')
        return '\n'.join(lines)

    def save(self):
        with open(self.header_filename, 'w') as fd:
            fd.write(self.render_header())

        with open(self.source_filename, 'w') as fd:
            fd.write(self.render_source())