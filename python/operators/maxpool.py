# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

def make_maxpool_argument_string(inputs, outputs):
    # inputs
    tensor = inputs[0]
    cname = tensor.sanitized_name
    shape = tensor.shape
    X = cname
    height = shape[1]
    width = shape[2]
    C_in = shape[3]

    # output
    tensor = outputs[0]
    Y = tensor.sanitized_name

    return f'{X}, {Y}, {height}, {width}, {C_in}'

class MaxPool2DDeep:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def render(self):
        argument_str = make_maxpool_argument_string(self.inputs, self.outputs)

        return f'maxpool2d_deep({argument_str});'
