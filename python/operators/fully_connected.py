# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

def make_fc_argument_string(inputs, outputs):
    # inputs
    for index, tensor in enumerate(inputs):
        cname = tensor.GetSanitizedName()
        shape = tensor.GetShape()
        if index == 0:
            X = cname
        elif index == 1:
            W = cname
            C_out = shape[0]
            C_in = shape[1]
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

    return f'{W}, {B}, {X}, {Y}, {C_out}, {C_in}, (uint16_t*) &{shifts}, (int16_t*) &{scales}'

class FullyConnectedDeepInShallowOutFinal:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def render(self):
        argument_str = make_fc_argument_string(self.inputs, self.outputs)

        return f'fc_deepin_shallowout_lin({argument_str});'
