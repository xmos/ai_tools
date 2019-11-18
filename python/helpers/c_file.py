# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import os
import struct
import re

class CFile():
    def __init__(self, name, includes=None, initializers=None, variables=None, functions=None, model=None):

        basename, extension = os.path.splitext(name)

        self.source_filename = f'{basename}.c'
        self.header_filename = f'{basename}.h'

        self.includes = includes or []
        self.initializers = initializers or []
        self.variables = variables or []
        self.functions = functions or []
        self.model = model
        
        self._build_macro_lookup()

    def get_filenames(self):
        return [
            self.header_filename,
            self.source_filename
        ]

    def _build_macro_lookup(self):
        self._macro_lookup = {}
        for initializer in self.initializers:
            name = initializer.sanitized_name
            identifier = name.upper()  #TODO: is this safe, might need to check that result
                                              #      is a valid macro
            replacement_list = None
            if self.model:
                stdtype = initializer.standard_type
                buffer = self.model.get_buffer(initializer.buffer, stdtype)

                replacement_list = ', '.join([str(v) for v in buffer])

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
            name = initializer.sanitized_name
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
            name = initializer.sanitized_name
            stdtype = initializer.standard_type
            shape = initializer.shape
            dims = ' * '.join([str(v) for v in shape])

            macro = self._macro_lookup[name]
            if macro['replacement-list']:
                identifier = macro['identifier']
                rhs = f' = {{{identifier}}}'
            else:
                rhs = ''
            lines.append(f'const {stdtype} WORD_ALIGNED {name}[{dims}]{rhs};')
        lines.append('')

        # variables
        for variable in self.variables:
            name = variable.sanitized_name
            stdtype = variable.standard_type
            shape = variable.shape
            dims = ' * '.join([str(v) for v in shape])
            lines.append(f'{stdtype} WORD_ALIGNED {name}[{dims}];')
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