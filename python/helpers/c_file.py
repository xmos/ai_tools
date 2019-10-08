# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import os

class CFile():
    def __init__(self, name, includes=None, initializers=None, functions=None):

        basename, extension = os.path.splitext(name)

        self.source_filename = f'{basename}.c'
        self.header_filename = f'{basename}.h'

        self.includes = includes or []
        self.initializers = initializers or []
        self.functions = functions or []
        
        self._build_macro_lookup()

    def _build_macro_lookup(self):
        self._macro_lookup = {}
        for initializer in self.initializers:
            name = initializer['name']
            macro = name.upper()  #TODO: is this safe, might need to check that result
                                              #      is a valid macro
            self._macro_lookup[name] = macro

    def add_include(self, include):
        self.includes.append(include)

    def add_initializer(self, initializer):
        self.initializers.append(initializer)
        self._build_macro_lookup()

    def add_function(self, funct):
        self.functions.append(funct)

    def render_header(self):
        lines = []

        for initializer in self.initializers:
            if len(initializer['values']):
                macro = self._macro_lookup[initializer['name']]
                values = ', '.join([str(v) for v in initializer['values']])
                #TODO: wrap these lines???
                lines.append(f'#define {macro} {values}')
        lines.append('')

        for function in self.functions:
            lines.append(function.render_declaration())
        lines.append('')

        lines.append('')
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
            name = initializer['name']
            ctype = initializer['type']
            dims = ' * '.join([str(v) for v in initializer['dims']])

            if len(initializer['values']):
                macro = self._macro_lookup[name]
                rhs = f' = {{{macro}}}'
            else:
                rhs = ''
            lines.append(f'const {ctype} WORD_ALIGNED {name}[{dims}]{rhs};')
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