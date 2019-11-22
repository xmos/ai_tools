# Copyright (c) 2019, XMOS Ltd, All rights reserved

import enum


class ValidOpCodes(enum.Enum):
    pass


class BuiltinOpCodes(ValidOpCodes):
    ADD = 0
    AVERAGE_POOL_2D = 1
    CONCATENATION = 2
    CONV_2D = 3
    DEPTHWISE_CONV_2D = 4
    DEPTH_TO_SPACE = 5
    DEQUANTIZE = 6
    EMBEDDING_LOOKUP = 7
    FLOOR = 8
    FULLY_CONNECTED = 9
    HASHTABLE_LOOKUP = 10
    L2_NORMALIZATION = 11
    L2_POOL_2D = 12
    LOCAL_RESPONSE_NORMALIZATION = 13
    LOGISTIC = 14
    LSH_PROJECTION = 15
    LSTM = 16
    MAX_POOL_2D = 17
    MUL = 18
    RELU = 19
    RELU_N1_TO_1 = 20
    RELU6 = 21
    RESHAPE = 22
    RESIZE_BILINEAR = 23
    RNN = 24
    SOFTMAX = 25
    SPACE_TO_DEPTH = 26
    SVDF = 27
    TANH = 28
    CONCAT_EMBEDDINGS = 29
    SKIP_GRAM = 30
    CALL = 31
    CUSTOM = 32
    EMBEDDING_LOOKUP_SPARSE = 33
    PAD = 34
    UNIDIRECTIONAL_SEQUENCE_RNN = 35
    GATHER = 36
    BATCH_TO_SPACE_ND = 37
    SPACE_TO_BATCH_ND = 38
    TRANSPOSE = 39
    MEAN = 40
    SUB = 41
    DIV = 42
    SQUEEZE = 43
    UNIDIRECTIONAL_SEQUENCE_LSTM = 44
    STRIDED_SLICE = 45
    BIDIRECTIONAL_SEQUENCE_RNN = 46
    EXP = 47
    TOPK_V2 = 48
    SPLIT = 49
    LOG_SOFTMAX = 50
    DELEGATE = 51
    BIDIRECTIONAL_SEQUENCE_LSTM = 52
    CAST = 53
    PRELU = 54
    MAXIMUM = 55
    ARG_MAX = 56
    MINIMUM = 57
    LESS = 58
    NEG = 59
    PADV2 = 60
    GREATER = 61
    GREATER_EQUAL = 62
    LESS_EQUAL = 63
    SELECT = 64
    SLICE = 65
    SIN = 66
    TRANSPOSE_CONV = 67
    SPARSE_TO_DENSE = 68
    TILE = 69
    EXPAND_DIMS = 70
    EQUAL = 71
    NOT_EQUAL = 72
    LOG = 73
    SUM = 74
    SQRT = 75
    RSQRT = 76
    SHAPE = 77
    POW = 78
    ARG_MIN = 79
    FAKE_QUANT = 80
    REDUCE_PROD = 81
    REDUCE_MAX = 82
    PACK = 83
    LOGICAL_OR = 84
    ONE_HOT = 85
    LOGICAL_AND = 86
    LOGICAL_NOT = 87
    UNPACK = 88
    REDUCE_MIN = 89
    FLOOR_DIV = 90
    REDUCE_ANY = 91
    SQUARE = 92
    ZEROS_LIKE = 93
    FILL = 94
    FLOOR_MOD = 95
    RANGE = 96
    RESIZE_NEAREST_NEIGHBOR = 97
    LEAKY_RELU = 98
    SQUARED_DIFFERENCE = 99
    MIRROR_PAD = 100
    ABS = 101
    SPLIT_V = 102
    UNIQUE = 103
    CEIL = 104
    REVERSE_V2 = 105
    ADD_N = 106
    GATHER_ND = 107
    COS = 108
    WHERE = 109
    RANK = 110
    ELU = 111
    REVERSE_SEQUENCE = 112
    MATRIX_DIAG = 113
    QUANTIZE = 114
    MATRIX_SET_DIAG = 115
    ROUND = 116
    HARD_SWISH = 117
    IF = 118
    WHILE = 119
    NON_MAX_SUPPRESSION_V4 = 120
    NON_MAX_SUPPRESSION_V5 = 121
    SCATTER_ND = 122


class XCOREOpCodes(ValidOpCodes):
    # TODO: consider an IntEnum for this instead of strings
    XC_argmax_16 = "XC_argmax_16"
    XC_maxpool2d_deep = "XC_maxpool2d_deep"
    XC_fc_deepin_shallowout_final = "XC_fc_deepin_shallowout_final"
    XC_conv2d_shallowin_deepout_relu = "XC_conv2d_shallowin_deepout_relu"
    XC_conv2d_deepin_deepout_relu = "XC_conv2d_deepin_deepout_relu"


class OperatorCode():
    def __init__(self, opcode, *, custom_opcode=None, version=None):
        assert isinstance(opcode, ValidOpCodes), "Invalid opcode!"
        self.version = version or 1

        if isinstance(opcode, XCOREOpCodes):
            self.builtin_opcode = BuiltinOpCodes.CUSTOM
            self.custom_opcode = opcode
        else:
            self.builtin_opcode = opcode
            if self.builtin_opcode == BuiltinOpCodes.CUSTOM:
                assert isinstance(custom_opcode, XCOREOpCodes), \
                    "Must provide custom_opcode if builtin_opcode is 'CUSTOM'!"
                self.custom_opcode = custom_opcode
            else:
                self.custom_opcode = None

    @property
    def opcode(self):
        return self.custom_opcode if self.builtin_opcode == BuiltinOpCodes.CUSTOM else self.builtin_opcode

    def __eq__(self, obj):
        return (isinstance(obj, OperatorCode)
                and obj.opcode == self.opcode
                and obj.version == self.version)

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        return f"{self.opcode.name} (version {self.version})"
