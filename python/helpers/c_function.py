# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

INDENT = '     ' # 5 spaces

class CFunction():
    def __init__(self, name, inputs=None, outputs=None, variables=None, operators=None):
        self.name = name
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.variables = variables or []
        self.operators = operators or []

    def add_operator(self, operator):
        self.operators.append(operator)

    def render_signature(self):
        return_type = 'void'  # assume void for now
        
        # inputs
        signature_inputs = []
        for i in self.inputs:
            name = i['name']
            ctype = i['type']
            signature_inputs.append(f'const {ctype} *{name}')
        signature_inputs = ', '.join(signature_inputs)

        # outputs
        signature_outputs = []
        for i in self.outputs:
            name = i['name']
            ctype = i['type']
            signature_outputs.append(f'{ctype} *{name}')
        signature_outputs = ', '.join(signature_outputs)

        #TODO: line wrap
        signature = f'{return_type} {self.name}({signature_inputs}, {signature_outputs})'
        
        return signature

    def render_declaration(self):
        signature = self.render_signature()
        
        return f'{signature};'

    def render_body(self):
        lines = []

        lines.append('{')

        # variables
        for variable in self.variables:
            name = variable['name']
            ctype = variable['type']
            dims = ' * '.join([str(v) for v in variable['dims']])
            lines.append(f'{INDENT}{ctype} WORD_ALIGNED {name}[{dims}];')
        lines.append('')

        for operator in self.operators:
            lines.append(f'{INDENT}{operator.render()}')

        lines.append('}')

        return '\n'.join(lines)

