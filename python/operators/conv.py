# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

def make_conv2d_argument_string(inputs, outputs, use_cin):
    # inputs
    for index, tensor in enumerate(inputs):
        cname = tensor.GetSanitizedName()
        shape = tensor.GetShape()
        if index == 0:
            X = cname
            height = shape[1]
            width = shape[2]
            C_in = shape[3]
        elif index == 1:
            K = cname
            C_out = shape[0]
            K_h = shape[1]
            K_w = shape[2]
        elif index == 2:
            B = cname
        elif index == 3:
            scales_offset = C_out
            shifts = f'{cname}[0]'
            scales = f'{cname}[{scales_offset}]'

    # output
    tensor = outputs[0]
    shape = tensor.GetShape()
    Y = tensor.GetSanitizedName()

    if use_cin:
        return f'{K}, (data16_t *){B}, {X}, {Y}, {height}, {width}, {K_h}, {K_w}, {C_out}, {C_in}, (int16_t*) &{shifts}, (int16_t*) &{scales}'
    else:
        return f'{K}, (data16_t *){B}, {X}, {Y}, {height}, {width}, {K_h}, {K_w}, {C_out}, (int16_t*) &{shifts}, (int16_t*) &{scales}'

class Conv2DShallowInDeepOut():
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def render(self):
        argument_str = make_conv2d_argument_string(self.inputs, self.outputs, use_cin=False)

        return f'//conv2d_shallowin_deepout_relu({argument_str});'

class Conv2DDeepInDeepOut():
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def render(self):
        argument_str = make_conv2d_argument_string(self.inputs, self.outputs, use_cin=True)

        return f'conv2d_deepin_deepout_relu({argument_str});'
