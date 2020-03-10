# Copyright (c) 2020, XMOS Ltd, All rights reserved

from .transformation_passes import *
from .lut_passes import (
    ReplaceTanhPass,
    ReplaceLogisticPass,
    ReplaceReLUPass,
    ReplaceReLU6Pass,
)
from .conv2d_passes import (
    Replace1x1Conv2dPass,
    ReplaceDepthwiseConv2dPass,
    ReplaceDeepinDeepoutConv2DPass,
    ReplaceShallowinDeepoutConv2DPass,
    ReplaceSingleinDeepoutDepthwiseConv2DPass
)
from .pooling_passes import (
    ReplaceMaxPool2DPass,
    ReplaceMaxPool2D2x2Pass,
    ReplaceAveragePool2DPass,
    ReplaceAveragePool2D2x2Pass,
    ReplaceGlobalAveragePool2DPass
)
