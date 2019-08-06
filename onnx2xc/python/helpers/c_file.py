# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import os

class CFile():
    def __init__(self, name, includes=None, variables=None, functions=None):

        basename, extension = os.path.splitext(name)

        self.source_filename = f'{basename}.c'
        self.header_filename = f'{basename}.h'

        self.includes = includes or []
        self.variables = variables or []
        self.functions = functions or []
        
        self._build_macro_lookup()

    def _build_macro_lookup(self):
        self._macro_lookup = {}
        for variable in self.variables:
            name = variable['name']
            macro = variable['name'].upper()  #TODO: is this safe, might need to check that result
                                              #      is a valid macro
            self._macro_lookup[name] = macro

    def add_include(self, include):
        self.includes.append(include)

    def add_variable(self, variable):
        self.variables.append(variable)
        self._build_macro_lookup()

    def add_function(self, funct):
        self.functions.append(funct)

    def render_header(self):
        lines = []

        for variable in self.variables:
            if len(variable['values']):
                macro = self._macro_lookup[variable['name']]
                values = ' '.join([str(v) for v in variable['values']])
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
            lines.append(f'#include "{include}""')
        lines.append(f'#include "{self.header_filename}"')
        lines.append('')

        # variables
        for variable in self.variables:
            name = variable['name']
            ctype = variable['type']
            dims = ' * '.join([str(v) for v in variable['dims']])
            if variable['const']:
                const_keyword = 'const'
            else:
                const_keyword = 'const'

            if len(variable['values']):
                macro = self._macro_lookup[name]
                rhs = f' = {macro}'
            else:
                rhs = ''
            lines.append(f'{const_keyword} {ctype} {name}[{dims}]{rhs};')
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