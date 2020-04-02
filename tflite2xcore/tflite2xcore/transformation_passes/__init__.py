# Copyright (c) 2020, XMOS Ltd, All rights reserved

from .transformation_passes import *  # TODO: fix this
from .argmax_passes import (
    AddArgMax16OutputPass,
    ReplaceArgMax16Pass
)
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
    ReplaceSingleinDeepoutDepthwiseConv2DPass,
    ParallelizeDIDOPass
)

from .fully_connected_passes import (
    ReplaceFullyConnectedOutputPass,
    ReplaceFullyConnectedIntermediatePass
)

from .pooling_passes import (
    ReplaceMaxPool2DPass,
    ReplaceMaxPool2D2x2Pass,
    ReplaceAveragePool2DPass,
    ReplaceAveragePool2D2x2Pass,
    ReplaceGlobalAveragePool2DPass
)
from .padding_passes import (
    FuseConv2dPaddingPass,
    SplitPaddingPass,
    FuseConsecutivePadsPass
)

from .quantize_dequantize_passes import (
    LegalizeQuantizedInputPass,
    LegalizeQuantizedOutputPass,
    LegalizeFloatInputPass,
    LegalizeFloatOutputPass,
    LegalizeQuantizeVersionPass,
)

from .cleanup_passes import (
    RemoveUnusedBuffersPass,
    RemoveDanglingTensorsPass
)

from .minification_passes import (
    LegalizeOutputTensorNamePass
)
