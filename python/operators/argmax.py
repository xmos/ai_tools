# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
from numpy import prod

def make_argmax_argument_string(inputs, outputs):
    # inputs
    tensor = inputs[0]
    N = prod(tensor.shape)
    A = tensor.sanitized_name

    # output
    tensor = outputs[0]
    C = tensor.sanitized_name

    return f'{A}, (int32_t *) {C}, {N}'


class ArgMax16:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def render(self):
        argument_str = make_argmax_argument_string(self.inputs, self.outputs)

        return f'argmax_16({argument_str});'
