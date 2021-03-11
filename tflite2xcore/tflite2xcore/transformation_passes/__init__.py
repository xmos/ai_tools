# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from .transformation_passes import ModelTransformationPass, CanonicalizeEmptyBuffersPass

from .lut_passes import (
    ReplaceTanhPass,
    ReplaceLogisticPass,
    ReplaceReLUPass,
    ReplaceReLU6Pass,
    LegalizeXCLookupTablePass,
)
from .conv2d_passes import (
    CanonicalizeSinglePixelConv2DPass,
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
    RemovePaddingInputPass,
    ReplacePadPass,
)

from .quantize_dequantize_passes import (
    RemoveRedundantInt8RequantizationPass,
    CanonicalizeQuantizedInputPass,
    CanonicalizeQuantizedOutputPass,
    CanonicalizeLceQuantizedOutputPass,
    CanonicalizeLceQuantizedInputPass,
    LegalizeFloatInputPass,
    LegalizeFloatOutputPass,
)

from .op_version_passes import LegalizeQuantizeVersionPass

from .dce_passes import (
    EliminateDeadOperatorsPass,
    EliminateDeadTensorsPass,
    EliminateDeadBuffersPass,
)

from .reshape_passes import (
    RemoveSubsequentReshapePass,
    RemovePrecedingReshapePass,
    CanonicalizeReshapePass,
)

from .renaming_passes import LegalizeOperatorOutputTensorNamePass

from .minification_passes import (
    MinifyQuantInfoPass,
    MinifyTensorNamesPass,
    UnifyEmptyBuffersPass,
)

from .word_alignment_passes import CanonicalizeConv2DInputChannels

from .parallelization_passes import (
    ParallelizeConv2dPass,
    ParallelizeDepthwiseConv2dPass,
    ParallelizeFullyConnectedPass,
    ParallelizeRequant16To8Pass,
    ParallelizePooling2DPass,
    ParallelizeGlobalAveragePool2DPass,
    ParallelizeBConv2dBinPass,
    ParallelizeBConv2dInt8Pass,
    ParallelizeLUTPass,
    ParallelizeAddPass,
)

from .scratch_memory_passes import (
    ScratchMemoryFullyConnectedPass,
    ScratchMemoryConv2dPass,
    ScratchMemoryConv2d1x1Pass,
    ScratchMemoryDepthwiseConv2dPass,
)

from .constant_propagation_passes import ConstantPropagationPass

from .lce_passes import (
    ReplaceBconv2DInt8Pass,
    ReplaceBconv2DInt8DeepInDeepOutPass,
    ReplaceBconv2DBitpackedPass,
    ReplaceBconv2DBitpackedDeepInPass,
    ReplaceLceQuantizePass,
    LegalizeXCBconv2DPaddingPass,
    LegalizeBconv2dInt8Pass,
    LegalizeBconv2dInt8DeepInDeepOutPass,
    LegalizeBconv2dBitpackedPass,
    LegalizeBconv2dBitpackedDeepInPass,
)

from .warning_passes import FloatingPointWarningPass

from .add_passes import ReplaceAddPass
