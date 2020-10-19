# Copyright (c) 2019, XMOS Ltd, All rights reserved

import enum
import aenum
import numpy as np  # type: ignore
from typing import Optional, Union, Any

from . import schema_py_generated as schema  # type: ignore


#  ----------------------------------------------------------------------------
#                                  TensorType
#  ----------------------------------------------------------------------------

TensorType = enum.IntEnum(
    "TensorType",
    {k: v for k, v in vars(schema.TensorType).items() if not k.startswith("__")},
)

__TensorType_to_stdint_type = {
    # TensorType.STRING: None,  # intentionally not supported
    TensorType.FLOAT64: "float64_t",
    TensorType.FLOAT32: "float32_t",
    TensorType.FLOAT16: "float16_t",
    TensorType.COMPLEX64: "complex64_t",
    TensorType.INT64: "int64_t",
    TensorType.INT32: "int32_t",
    TensorType.INT16: "int16_t",
    TensorType.INT8: "int8_t",
    TensorType.UINT8: "uint8_t",
    TensorType.BOOL: "uint8_t",
}
TensorType.to_stdint_type = lambda self: __TensorType_to_stdint_type[self]

__TensorType_to_bytes = {
    # TensorType.STRING: None,  # intentionally not supported
    TensorType.FLOAT64: 8,
    TensorType.FLOAT32: 4,
    TensorType.FLOAT16: 2,
    TensorType.COMPLEX64: 8,
    TensorType.INT64: 8,
    TensorType.INT32: 4,
    TensorType.INT16: 2,
    TensorType.INT8: 1,
    TensorType.UINT8: 1,
    TensorType.BOOL: 1,
}
TensorType.to_bytes = lambda self: __TensorType_to_bytes[self]

__TensorType_to_numpy_dtype = {
    # TensorType.STRING: None,  # intentionally not supported
    TensorType.FLOAT64: np.float64,
    TensorType.FLOAT32: np.float32,
    TensorType.FLOAT16: np.float16,
    TensorType.COMPLEX64: np.complex64,
    TensorType.INT64: np.int64,
    TensorType.INT32: np.int32,
    TensorType.INT16: np.int16,
    TensorType.INT8: np.int8,
    TensorType.UINT8: np.uint8,
    TensorType.BOOL: np.bool_,
}
TensorType.to_numpy_dtype = lambda self: __TensorType_to_numpy_dtype[self]

__TensorType_from_numpy_dtype = {
    np.dtype(np.float64): TensorType.FLOAT64,
    np.dtype(np.float32): TensorType.FLOAT32,
    np.dtype(np.float16): TensorType.FLOAT16,
    np.dtype(np.complex64): TensorType.COMPLEX64,
    np.dtype(np.int64): TensorType.INT64,
    np.dtype(np.int32): TensorType.INT32,
    np.dtype(np.int16): TensorType.INT16,
    np.dtype(np.int8): TensorType.INT8,
    np.dtype(np.uint8): TensorType.UINT8,
    np.dtype(np.bool_): TensorType.BOOL,
}
TensorType.from_numpy_dtype = lambda x: __TensorType_from_numpy_dtype[np.dtype(x)]


#  ----------------------------------------------------------------------------
#                               Operator Codes
#  ----------------------------------------------------------------------------


class ValidOpCodes:
    pass


class KnownOpCodes(ValidOpCodes, enum.Enum):
    pass


BuiltinOpCodes = KnownOpCodes(
    "BuiltinOpCodes",
    {k: v for k, v in vars(schema.BuiltinOperator).items() if not k.startswith("__")},
)


class CustomOpCodes(ValidOpCodes):
    pass


class ExternalOpCodes(CustomOpCodes, aenum.Enum):
    @classmethod
    def add_new_opcode(cls, name: str) -> "ExternalOpCodes":
        assert name.isidentifier()
        try:
            return cls[name]
        except KeyError:
            aenum.extend_enum(cls, name)
            return cls[name]


class XCOREOpCodes(CustomOpCodes, KnownOpCodes):
    # TODO: consider an IntEnum for this instead of strings
    XC_lookup_8 = "XC_lookup_8"
    XC_argmax_16 = "XC_argmax_16"  # currently not used by any passes
    XC_maxpool2d = "XC_maxpool2d"
    XC_avgpool2d = "XC_avgpool2d"
    XC_avgpool2d_global = "XC_avgpool2d_global"
    XC_fc = "XC_fc"
    XC_requantize_16_to_8 = "XC_requantize_16_to_8"  # currently unused
    XC_conv2d_shallowin = "XC_conv2d_shallowin"
    XC_conv2d_deep = "XC_conv2d_deep"
    XC_conv2d_1x1 = "XC_conv2d_1x1"
    XC_conv2d_depthwise = "XC_conv2d_depthwise"
    XC_bsign_8 = "XC_bsign_8"
    XC_pad = "XC_pad"
    XC_bconv2d_int8 = "XC_bconv2d_int8"
    XC_bconv2d_int8_DIDO = "XC_bconv2d_int8_DIDO"
    XC_bconv2d_bin = "XC_bconv2d_bin"
    XC_bconv2d_bin_DI = "XC_bconv2d_bin_DI"


class OperatorCode:
    def __init__(self, opcode: ValidOpCodes, *, version: Optional[int] = None) -> None:
        self.version = version or 1
        self.code = opcode

    @property
    def name(self) -> str:
        return self.code.name

    @property
    def value(self) -> Union[int, str]:
        return self.code.value

    def __eq__(self, obj: Any) -> bool:
        return (
            isinstance(obj, OperatorCode)
            and obj.code is self.code
            and obj.version == self.version
        )

    def __hash__(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        return f"{self.code} (version {self.version})"


#  ----------------------------------------------------------------------------
#                               Builtin Options
#  ----------------------------------------------------------------------------

BuiltinOptions = enum.Enum(
    "BuiltinOptions",
    {k: v for k, v in vars(schema.BuiltinOptions).items() if not k.startswith("__")},
)

# this mapping should follow the schema and:
# tensorflow/tensorflow/lite/core/api/flatbuffer_conversions.cc
__BuiltinOpCodes_to_BuiltinOptions = {
    BuiltinOpCodes.ADD: BuiltinOptions.AddOptions,
    BuiltinOpCodes.AVERAGE_POOL_2D: BuiltinOptions.Pool2DOptions,
    BuiltinOpCodes.CONCATENATION: BuiltinOptions.ConcatenationOptions,
    BuiltinOpCodes.CONV_2D: BuiltinOptions.Conv2DOptions,
    BuiltinOpCodes.DEPTHWISE_CONV_2D: BuiltinOptions.DepthwiseConv2DOptions,
    BuiltinOpCodes.DEPTH_TO_SPACE: BuiltinOptions.DepthToSpaceOptions,
    BuiltinOpCodes.DEQUANTIZE: BuiltinOptions.DequantizeOptions,
    BuiltinOpCodes.EMBEDDING_LOOKUP: BuiltinOptions.NONE,
    BuiltinOpCodes.FLOOR: BuiltinOptions.NONE,
    BuiltinOpCodes.FULLY_CONNECTED: BuiltinOptions.FullyConnectedOptions,
    BuiltinOpCodes.HASHTABLE_LOOKUP: BuiltinOptions.NONE,
    BuiltinOpCodes.L2_NORMALIZATION: BuiltinOptions.L2NormOptions,
    BuiltinOpCodes.L2_POOL_2D: BuiltinOptions.Pool2DOptions,
    BuiltinOpCodes.LOCAL_RESPONSE_NORMALIZATION: BuiltinOptions.LocalResponseNormalizationOptions,
    BuiltinOpCodes.LOGISTIC: BuiltinOptions.NONE,
    BuiltinOpCodes.LSH_PROJECTION: BuiltinOptions.LSHProjectionOptions,
    BuiltinOpCodes.LSTM: BuiltinOptions.LSTMOptions,
    BuiltinOpCodes.MAX_POOL_2D: BuiltinOptions.Pool2DOptions,
    BuiltinOpCodes.MUL: BuiltinOptions.MulOptions,
    BuiltinOpCodes.RELU: BuiltinOptions.NONE,
    BuiltinOpCodes.RELU_N1_TO_1: BuiltinOptions.NONE,
    BuiltinOpCodes.RELU6: BuiltinOptions.NONE,
    BuiltinOpCodes.RESHAPE: BuiltinOptions.ReshapeOptions,
    BuiltinOpCodes.RESIZE_BILINEAR: BuiltinOptions.ResizeBilinearOptions,
    BuiltinOpCodes.RNN: BuiltinOptions.RNNOptions,
    BuiltinOpCodes.SOFTMAX: BuiltinOptions.SoftmaxOptions,
    BuiltinOpCodes.SPACE_TO_DEPTH: BuiltinOptions.SpaceToDepthOptions,
    BuiltinOpCodes.SVDF: BuiltinOptions.SVDFOptions,
    BuiltinOpCodes.TANH: BuiltinOptions.NONE,
    BuiltinOpCodes.CONCAT_EMBEDDINGS: BuiltinOptions.ConcatEmbeddingsOptions,
    BuiltinOpCodes.SKIP_GRAM: BuiltinOptions.SkipGramOptions,
    BuiltinOpCodes.CALL: BuiltinOptions.CallOptions,
    BuiltinOpCodes.CUSTOM: BuiltinOptions.NONE,
    BuiltinOpCodes.EMBEDDING_LOOKUP_SPARSE: BuiltinOptions.EmbeddingLookupSparseOptions,
    BuiltinOpCodes.PAD: BuiltinOptions.PadOptions,
    BuiltinOpCodes.UNIDIRECTIONAL_SEQUENCE_RNN: BuiltinOptions.SequenceRNNOptions,
    BuiltinOpCodes.GATHER: BuiltinOptions.GatherOptions,
    BuiltinOpCodes.BATCH_TO_SPACE_ND: BuiltinOptions.BatchToSpaceNDOptions,
    BuiltinOpCodes.SPACE_TO_BATCH_ND: BuiltinOptions.SpaceToBatchNDOptions,
    BuiltinOpCodes.TRANSPOSE: BuiltinOptions.TransposeOptions,
    BuiltinOpCodes.MEAN: BuiltinOptions.ReducerOptions,
    BuiltinOpCodes.SUB: BuiltinOptions.SubOptions,
    BuiltinOpCodes.DIV: BuiltinOptions.DivOptions,
    BuiltinOpCodes.SQUEEZE: BuiltinOptions.SqueezeOptions,
    BuiltinOpCodes.UNIDIRECTIONAL_SEQUENCE_LSTM: BuiltinOptions.UnidirectionalSequenceLSTMOptions,
    BuiltinOpCodes.STRIDED_SLICE: BuiltinOptions.StridedSliceOptions,
    BuiltinOpCodes.BIDIRECTIONAL_SEQUENCE_RNN: BuiltinOptions.BidirectionalSequenceRNNOptions,
    BuiltinOpCodes.EXP: BuiltinOptions.ExpOptions,
    BuiltinOpCodes.TOPK_V2: BuiltinOptions.TopKV2Options,
    BuiltinOpCodes.SPLIT: BuiltinOptions.SplitOptions,
    BuiltinOpCodes.LOG_SOFTMAX: BuiltinOptions.LogSoftmaxOptions,
    BuiltinOpCodes.DELEGATE: BuiltinOptions.NONE,
    BuiltinOpCodes.BIDIRECTIONAL_SEQUENCE_LSTM: BuiltinOptions.BidirectionalSequenceLSTMOptions,
    BuiltinOpCodes.CAST: BuiltinOptions.CastOptions,
    BuiltinOpCodes.PRELU: BuiltinOptions.NONE,
    BuiltinOpCodes.MAXIMUM: BuiltinOptions.MaximumMinimumOptions,
    BuiltinOpCodes.ARG_MAX: BuiltinOptions.ArgMaxOptions,
    BuiltinOpCodes.MINIMUM: BuiltinOptions.MaximumMinimumOptions,
    BuiltinOpCodes.LESS: BuiltinOptions.LessOptions,
    BuiltinOpCodes.NEG: BuiltinOptions.NegOptions,
    BuiltinOpCodes.PADV2: BuiltinOptions.PadV2Options,
    BuiltinOpCodes.GREATER: BuiltinOptions.GreaterOptions,
    BuiltinOpCodes.GREATER_EQUAL: BuiltinOptions.GreaterEqualOptions,
    BuiltinOpCodes.LESS_EQUAL: BuiltinOptions.LessEqualOptions,
    BuiltinOpCodes.SELECT: BuiltinOptions.SelectOptions,
    BuiltinOpCodes.SLICE: BuiltinOptions.SliceOptions,
    BuiltinOpCodes.SIN: BuiltinOptions.NONE,
    BuiltinOpCodes.TRANSPOSE_CONV: BuiltinOptions.TransposeConvOptions,
    BuiltinOpCodes.SPARSE_TO_DENSE: BuiltinOptions.SparseToDenseOptions,
    BuiltinOpCodes.TILE: BuiltinOptions.TileOptions,
    BuiltinOpCodes.EXPAND_DIMS: BuiltinOptions.ExpandDimsOptions,
    BuiltinOpCodes.EQUAL: BuiltinOptions.EqualOptions,
    BuiltinOpCodes.NOT_EQUAL: BuiltinOptions.NotEqualOptions,
    BuiltinOpCodes.LOG: BuiltinOptions.NONE,
    BuiltinOpCodes.SUM: BuiltinOptions.ReducerOptions,
    BuiltinOpCodes.SQRT: BuiltinOptions.NONE,
    BuiltinOpCodes.RSQRT: BuiltinOptions.NONE,
    BuiltinOpCodes.SHAPE: BuiltinOptions.ShapeOptions,
    BuiltinOpCodes.POW: BuiltinOptions.PowOptions,
    BuiltinOpCodes.ARG_MIN: BuiltinOptions.ArgMinOptions,
    BuiltinOpCodes.FAKE_QUANT: BuiltinOptions.FakeQuantOptions,
    BuiltinOpCodes.REDUCE_PROD: BuiltinOptions.ReducerOptions,
    BuiltinOpCodes.REDUCE_MAX: BuiltinOptions.ReducerOptions,
    BuiltinOpCodes.PACK: BuiltinOptions.PackOptions,
    BuiltinOpCodes.LOGICAL_OR: BuiltinOptions.LogicalOrOptions,
    BuiltinOpCodes.ONE_HOT: BuiltinOptions.OneHotOptions,
    BuiltinOpCodes.LOGICAL_AND: BuiltinOptions.LogicalAndOptions,
    BuiltinOpCodes.LOGICAL_NOT: BuiltinOptions.LogicalNotOptions,
    BuiltinOpCodes.UNPACK: BuiltinOptions.UnpackOptions,
    BuiltinOpCodes.REDUCE_MIN: BuiltinOptions.ReducerOptions,
    BuiltinOpCodes.FLOOR_DIV: BuiltinOptions.FloorDivOptions,
    BuiltinOpCodes.REDUCE_ANY: BuiltinOptions.ReducerOptions,
    BuiltinOpCodes.SQUARE: BuiltinOptions.SquareOptions,
    BuiltinOpCodes.ZEROS_LIKE: BuiltinOptions.ZerosLikeOptions,
    BuiltinOpCodes.FILL: BuiltinOptions.FillOptions,
    BuiltinOpCodes.FLOOR_MOD: BuiltinOptions.FloorModOptions,
    BuiltinOpCodes.RANGE: BuiltinOptions.RangeOptions,
    BuiltinOpCodes.RESIZE_NEAREST_NEIGHBOR: BuiltinOptions.ResizeNearestNeighborOptions,
    BuiltinOpCodes.LEAKY_RELU: BuiltinOptions.LeakyReluOptions,
    BuiltinOpCodes.SQUARED_DIFFERENCE: BuiltinOptions.SquaredDifferenceOptions,
    BuiltinOpCodes.MIRROR_PAD: BuiltinOptions.MirrorPadOptions,
    BuiltinOpCodes.ABS: BuiltinOptions.AbsOptions,
    BuiltinOpCodes.SPLIT_V: BuiltinOptions.SplitVOptions,
    BuiltinOpCodes.UNIQUE: BuiltinOptions.UniqueOptions,
    BuiltinOpCodes.CEIL: BuiltinOptions.NONE,
    BuiltinOpCodes.REVERSE_V2: BuiltinOptions.ReverseV2Options,
    BuiltinOpCodes.ADD_N: BuiltinOptions.AddNOptions,
    BuiltinOpCodes.GATHER_ND: BuiltinOptions.GatherNdOptions,
    BuiltinOpCodes.COS: BuiltinOptions.CosOptions,
    BuiltinOpCodes.WHERE: BuiltinOptions.WhereOptions,
    BuiltinOpCodes.RANK: BuiltinOptions.RankOptions,
    BuiltinOpCodes.ELU: BuiltinOptions.NONE,
    BuiltinOpCodes.REVERSE_SEQUENCE: BuiltinOptions.ReverseSequenceOptions,
    BuiltinOpCodes.MATRIX_DIAG: BuiltinOptions.MatrixDiagOptions,
    BuiltinOpCodes.QUANTIZE: BuiltinOptions.QuantizeOptions,
    BuiltinOpCodes.MATRIX_SET_DIAG: BuiltinOptions.MatrixSetDiagOptions,
    BuiltinOpCodes.ROUND: BuiltinOptions.NONE,
    BuiltinOpCodes.HARD_SWISH: BuiltinOptions.HardSwishOptions,
    BuiltinOpCodes.IF: BuiltinOptions.IfOptions,
    BuiltinOpCodes.WHILE: BuiltinOptions.WhileOptions,
    BuiltinOpCodes.NON_MAX_SUPPRESSION_V4: BuiltinOptions.NonMaxSuppressionV4Options,
    BuiltinOpCodes.NON_MAX_SUPPRESSION_V5: BuiltinOptions.NonMaxSuppressionV5Options,
    BuiltinOpCodes.SCATTER_ND: BuiltinOptions.ScatterNdOptions,
    BuiltinOpCodes.SELECT_V2: BuiltinOptions.SelectV2Options,
    BuiltinOpCodes.DENSIFY: BuiltinOptions.DensifyOptions,
    BuiltinOpCodes.SEGMENT_SUM: BuiltinOptions.SegmentSumOptions,
    BuiltinOpCodes.BATCH_MATMUL: BuiltinOptions.BatchMatMulOptions,
}
BuiltinOpCodes.to_BuiltinOptions = lambda self: __BuiltinOpCodes_to_BuiltinOptions[self]

#  ----------------------------------------------------------------------------
#                               Misc Enums
#  ----------------------------------------------------------------------------


ActivationFunctionType = enum.Enum(
    "ActivationFunctionType",
    {
        k: v
        for k, v in vars(schema.ActivationFunctionType).items()
        if not k.startswith("__")
    },
)


QuantizationDetails = enum.Enum(
    "QuantizationDetails",
    {
        k: v
        for k, v in vars(schema.QuantizationDetails).items()
        if not k.startswith("__")
    },
)


FullyConnectedOptionsWeightsFormat = enum.Enum(
    "FullyConnectedOptionsWeightsFormat",
    {
        k: v
        for k, v in vars(schema.FullyConnectedOptionsWeightsFormat).items()
        if not k.startswith("__")
    },
)


#  ----------------------------------------------------------------------------
#                               Padding
#  ----------------------------------------------------------------------------


Padding = enum.Enum(
    "Padding", {k: v for k, v in vars(schema.Padding).items() if not k.startswith("__")}
)
