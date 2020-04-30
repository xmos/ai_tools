# Copyright (c) 2020, XMOS Ltd, All rights reserved

from .transformation_passes import *  # TODO: fix this

from .lut_passes import (
    ReplaceTanhPass,
    ReplaceLogisticPass,
    ReplaceReLUPass,
    ReplaceReLU6Pass,
    LegalizeXCLookupTablePass,
)
from .conv2d_passes import (
    CanonicalizeSingleinDepthwiseConv2DPass,
    LegalizeSingleinConv2DPass,
    Replace1x1Conv2dPass,
    LegalizeXC1x1ConvPass,
    ReplaceDepthwiseConv2dPass,
    LegalizeXCDepthwiseConvPass,
    ReplaceDeepConv2dPass,
    LegalizeXCDeepConvPass,
    ReplaceShallowinConv2dPass,
    LegalizeXCShallowinConvPass,
    ParallelizeXCConv2dPass,
    ParallelizeDeepConv2dPass,
)

from .fully_connected_passes import (
    ReplaceFullyConnectedPass,
    LegalizeXCFullyConnectedPass,
)

from .pooling_passes import (
    ReplaceMaxPool2DPass,
    ReplaceMaxPool2D2x2Pass,
    ReplaceAveragePool2DPass,
    ReplaceAveragePool2D2x2Pass,
    ReplaceGlobalAveragePool2DPass,
)
from .padding_passes import (
    FuseConv2dPaddingPass,
    SplitPaddingPass,
    FuseConsecutivePadsPass,
)

from .quantize_dequantize_passes import (
    CanonicalizeQuantizedInputPass,
    CanonicalizeQuantizedOutputPass,
    LegalizeFloatInputPass,
    LegalizeFloatOutputPass,
)

from .op_version_passes import LegalizeQuantizeVersionPass

from .dce_passes import (
    EliminateDeadOperatorsPass,
    EliminateDeadTensorsPass,
    EliminateDeadBuffersPass,
)

from .renaming_passes import LegalizeOperatorOutputTensorNamePass

from .minification_passes import MinifyQuantInfoPass, MinifyTensorNamesPass

from .word_alignment_passes import CanonicalizeConv2DInputChannels
