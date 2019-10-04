# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

def make_fc_argument_string(inputs, outputs):
    # inputs
    for index, tensor in enumerate(inputs):
        cname = tensor.GetSanitizedName()
        shape = tensor.GetShape()
        if index == 0:
            X = cname
            C_in = shape[-1]
        elif index == 1:
            W = cname
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

    return f'{W}, {B}, {X}, {Y}, {C_out}, {C_in}, {shifts}, {scales}'

class FullyConnectedDeepInShallowOutFinal:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def render(self):
        argument_str = make_fc_argument_string(self.inputs, self.outputs)

        return f'fc_deepin_shallowout_lin({argument_str});'
