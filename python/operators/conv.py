# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

def make_conv2d_argument_string(inputs, outputs):
    # inputs
    for index, tensor in enumerate(inputs):
        cname = tensor.GetSanitizedName()
        shape = tensor.GetShape()
        if index == 0:
            X = cname
            height = shape[1]
            width = shape[2]
        elif index == 1:
            K = cname
            K_h = shape[1]
            K_w = shape[2]
        elif index == 2:
            B = cname
        elif index == 3:
            shifts = f'{cname}[0]'
            scales = f'{cname}[1]'

    # output
    tensor = outputs[0]
    shape = tensor.GetShape()
    Y = tensor.GetSanitizedName()
    C_out = shape[-1]

    return f'{K}, {B}, {X}, {Y}, {height}, {width}, {K_h}, {K_w}, {C_out}, {shifts}, {scales}'

class Conv2DShallowInDeepOut():
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def render(self):
        argument_str = make_conv2d_argument_string(self.inputs, self.outputs)

        return f'conv2d_shallowin_deepout_relu({argument_str});'

class Conv2DDeepInDeepOut():
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def render(self):
        argument_str = make_conv2d_argument_string(self.inputs, self.outputs)

        return f'conv2d_deepin_deepout_relu({argument_str});'
