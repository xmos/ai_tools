# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

def make_conv2d_shallowin_deepout_argument_string(inputs, outputs, model):
    # inputs
    shifts_scales_tensor_name = None

    for index, tensor in enumerate(inputs):
        cname = tensor.sanitized_name
        shape = tensor.shape
        if index == 0:
            X = cname
            height = shape[1]
            width = shape[2]
        elif index == 1:
            K = cname
        elif index == 2:
            B = cname
        elif index == 3:
            shifts_scales_tensor_name = cname
        elif index == 4:
            data = tensor.buffer.unpack('int32_t')
            C_out = data[0]
            K_h = data[1]
            K_w = data[2]

    scales_offset = C_out
    shifts = f'{shifts_scales_tensor_name}[0]'
    scales = f'{shifts_scales_tensor_name}[{scales_offset}]'

    # output
    tensor = outputs[0]
    shape = tensor.shape
    Y = tensor.sanitized_name

    return f'{K}, (data16_t *){B}, (int8_t*){X}, {Y}, {height}, {width}, {K_h}, {K_w}, {C_out}, (int16_t*) &{shifts}, (int16_t*) &{scales}'

def make_conv2d_deepin_deepout_argument_string(inputs, outputs):
    # inputs
    for index, tensor in enumerate(inputs):
        cname = tensor.sanitized_name
        shape = tensor.shape
        if index == 0:
            X = cname
            height = shape[1]
            width = shape[2]
        elif index == 1:
            K = cname
            C_out = shape[0] * shape[4]
            C_in = shape[3] * shape[5]
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
    shape = tensor.shape
    Y = tensor.sanitized_name

    return f'{K}, (data16_t *){B}, (int8_t *){X}, {Y}, {height}, {width}, {K_h}, {K_w}, {C_out}, {C_in}, (int16_t*) &{shifts}, (int16_t*) &{scales}'

class Conv2DShallowInDeepOut():
    def __init__(self, inputs, outputs, model):
        self.inputs = inputs
        self.outputs = outputs
        self.model = model

    def render(self):
        argument_str = make_conv2d_shallowin_deepout_argument_string(self.inputs, self.outputs, self.model)

        return f'conv2d_shallowin_deepout_relu({argument_str});'

class Conv2DDeepInDeepOut():
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def render(self):
        argument_str = make_conv2d_deepin_deepout_argument_string(self.inputs, self.outputs)

        return f'conv2d_deepin_deepout_relu({argument_str});'
