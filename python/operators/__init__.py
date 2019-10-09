# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

from operators import conv
from operators import fully_connected
from operators import maxpool
from operators import argmax

class UnsupportedOperator(Exception):
    def __init__(self, name):
        super().__init__(f'Operator {name} not supported.')

class UnimplementedOperator(Exception):
    def __init__(self, name):
        super().__init__(f'Operator {name} not implemented.')

def create(name, inputs=None, outputs=None):
    if name == 'XC_conv2d_shallowin_deepout_relu':
        return conv.Conv2DShallowInDeepOut(inputs, outputs)
    elif name == 'XC_conv2d_deepin_deepout_relu':
        return conv.Conv2DDeepInDeepOut(inputs, outputs)
    elif name == 'XC_fc_deepin_shallowout_final':
        return fully_connected.FullyConnectedDeepInShallowOutFinal(inputs, outputs)
    elif name == 'XC_maxpool2d_deep':
        return maxpool.MaxPool2DDeep(inputs, outputs)
    elif name == 'XC_argmax_16':
        return argmax.ArgMax16(inputs, outputs)
    else:
        raise UnsupportedOperator(name)

