# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

INDENT = '     ' # 5 spaces

class CFunction():
    def __init__(self, name, inputs=None, outputs=None, variables=None, operators=None):
        self.name = name
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.variables = variables or []
        self.operators = operators or []

        self._built_typedefs()

    def _built_typedefs(self):
        self.input_typedefs = []
        self.output_typedefs = []

        # inputs
        for tensor in self.inputs:
            name = tensor.sanitized_name
            lower_name = name.lower()
            typedef = {
                'stdtype': tensor.standard_type,
                'type_identifier': f'{lower_name}_t',
                'variable_identifier': name,
                'dimensions': ' * '.join([str(v) for v in tensor.shape])

            }
            self.input_typedefs.append(typedef)

        # outputs
        for tensor in self.outputs:
            name = tensor.sanitized_name
            lower_name = name.lower()
            typedef = {
                'stdtype': tensor.standard_type,
                'type_identifier': f'{lower_name}_t',
                'variable_identifier': name,
                'dimensions': ' * '.join([str(v) for v in tensor.shape])

            }
            self.output_typedefs.append(typedef)

    def add_operator(self, operator):
        self.operators.append(operator)

    def render_typedefs(self):
        lines = []
        for typedef in self.input_typedefs+self.output_typedefs:
            stdtype = typedef['stdtype']
            type_identifier = typedef['type_identifier']
            dimensions = typedef['dimensions']
            line = f'typedef {stdtype} {type_identifier}[{dimensions}];'
            lines.append(line)

        return '\n'.join(lines)

    def render_signature(self):
        return_type = 'void'  # assume void for now
        
        # inputs
        signature_inputs = []
        for typedef in self.input_typedefs:
            type_identifier = typedef['type_identifier']
            variable_identifier = typedef['variable_identifier']
            signature_inputs.append(f'const {type_identifier} *{variable_identifier}')
        signature_inputs = ', '.join(signature_inputs)

        # outputs
        signature_outputs = []
        for typedef in self.output_typedefs:
            type_identifier = typedef['type_identifier']
            variable_identifier = typedef['variable_identifier']
            signature_outputs.append(f'{type_identifier} *{variable_identifier}')
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
            name = variable.Sanitized_name
            stdtype = variable.standard_type
            shape = variable.shape
            dims = ' * '.join([str(v) for v in shape])
            lines.append(f'{INDENT}{stdtype} WORD_ALIGNED {name}[{dims}];')
        lines.append('')

        for operator in self.operators:
            lines.append(f'{INDENT}{operator.render()}')

        lines.append('}')

        return '\n'.join(lines)

