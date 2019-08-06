# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import operators

class Conv:
    def __init__(self, inputs, outputs, comment=None):
        self.inputs = inputs
        self.outputs = outputs
        self.comment = comment

    def render(self):
        inputs = ', '.join(self.inputs)
        outputs = ', '.join(self.outputs)
        return f'convolution({inputs}, {outputs}); # {self.comment}'
