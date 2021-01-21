# Copyright (c) 2020, XMOS Ltd, All rights reserved

import numpy as np
from copy import deepcopy
from typing import Tuple, Optional, Dict, Any

from tflite2xcore.transformation_passes.lce_passes import (
    ReplaceBconv2DPass,
    XC_BCONV2D_OPCODES,
)
from tflite2xcore.transformation_passes import ModelTransformationPass
from tflite2xcore.xcore_schema import (
    XCOREModel,
    Subgraph,
    TensorType,
    ActivationFunctionType,
    Padding,
    OperatorCode,
    ValidOpCodes,
    ExternalOpCodes,
    XCOREOpCodes,
)
from tflite2xcore.utils import calculate_same_output_size, calculate_valid_output_size

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    generate_dummy_data,
)

from ..conftest import ParamsType
from ..conftest import (  # pylint: disable=unused-import
    _make_name_type_pairs,
    _test_non_matching_params,
    test_matching_params,
    test_non_matching_tensors,
    test_replace_mutate as _test_mutate,
)
from ..test_conv2d_passes.conftest import test_replace_mutate as test_conv2d_mutate
from ..test_conv2d_passes.conftest import (  # pylint: disable=unused-import
    PARAMS,
    test_non_matching_input_channels,
    test_non_matching_output_channels,
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------


PARAMS = deepcopy(PARAMS)


def update_lce_params(PARAMS: ParamsType) -> ParamsType:
    for key in (
        "input_channels",
        "non_matching_input_channels",
        "output_channels",
        "non_matching_output_channels",
    ):
        PARAMS["default"][key] = PARAMS["extended"][key][:-1]
        PARAMS["smoke"][key] = PARAMS["default"][key][:-1]
    non_matching_tensors = PARAMS["extended"]["non_matching_tensors"][::2]
    PARAMS["default"]["non_matching_tensors"] = non_matching_tensors
    PARAMS["smoke"]["non_matching_tensors"] = non_matching_tensors

    return PARAMS


PARAMS["extended"].update(
    {"input_channels": [32, 128, 256], "non_matching_input_channels": [48, 130, 245]}
)

#  ----------------------------------------------------------------------------
#                              MODEL BUILDERS
#  ----------------------------------------------------------------------------


def build_LceQuantize(
    subgraph: Optional[Subgraph] = None,
    *,
    input_shape: Tuple[int, int, int],
    input_tensor_type: TensorType = TensorType.INT8,
) -> XCOREModel:
    subgraph = subgraph or Subgraph(model=XCOREModel())
    height, width, channels = input_shape
    input_shape = (1, height, width, channels)
    output_shape = (1, height, width, int(np.ceil(channels / 32)))

    tin = subgraph.create_tensor("input", input_tensor_type, input_shape, isinput=True)
    tout = subgraph.create_tensor(
        "output", TensorType.INT32, output_shape, isoutput=True,
    )

    subgraph.create_operator(
        OperatorCode(ExternalOpCodes.LceQuantize), inputs=[tin], outputs=[tout]
    )

    return subgraph.model


def build_bconv2d(
    subgraph: Optional[Subgraph] = None,
    *,
    weight_shape: Tuple[int, int, int, int],
    input_size: Tuple[int, int],
    padding: Optional[Padding],
    strides: Tuple[int, int],
    opcode: ValidOpCodes,
    output_tensor_type: TensorType = TensorType.INT8,
) -> XCOREModel:
    subgraph = subgraph or Subgraph(model=XCOREModel())

    # the given shapes are not bitpacked (i.e. true channel counts)
    # so we bitpack them
    C_out, _, _, C_in = weight_shape
    bitpacked_input_channels = int(np.ceil(C_in / 32))
    weight_shape = (*weight_shape[:3], bitpacked_input_channels)

    # create input tensors
    input_shape = [1, *input_size, bitpacked_input_channels]
    tin = subgraph.create_tensor("input", TensorType.INT32, input_shape, isinput=True)

    w = subgraph.create_tensor("weights", TensorType.INT32, weight_shape)
    w.buffer.data = generate_dummy_data(w.shape, np.int32)

    input_tensors = [tin, w]
    if output_tensor_type is TensorType.INT32:
        output_threshold = subgraph.create_tensor(
            "output_threshold", TensorType.INT32, weight_shape[:1]
        )
        output_threshold.buffer.data = generate_dummy_data(
            output_threshold.shape, np.int32
        )

        input_tensors.append(output_threshold)

        output_quantization = None
    elif output_tensor_type is TensorType.INT8:
        post_act_params: Dict[str, Any] = {"shape": weight_shape[:1]}
        if opcode in XC_BCONV2D_OPCODES:
            post_act_params["type_"] = TensorType.INT16
            dummy_data = generate_dummy_data(post_act_params["shape"], np.int16)
        else:
            post_act_params["type_"] = TensorType.FLOAT32
            dummy_data = generate_dummy_data(post_act_params["shape"], np.float32)

        post_act_mult = subgraph.create_tensor("post_act_mult", **post_act_params)
        post_act_mult.buffer.data = dummy_data

        post_act_bias = subgraph.create_tensor("post_act_bias", **post_act_params)
        post_act_bias.buffer.data = dummy_data

        input_tensors.extend([post_act_mult, post_act_bias])

        output_quantization = {"scale": [0.46], "zero_point": [-54]}
    else:
        raise ValueError(
            f"output_tensor_type must be {TensorType.INT32} or {TensorType.INT8}"
        )

    # check padding and determine output size
    if padding is Padding.SAME:
        output_size = calculate_same_output_size(input_size, strides)
    else:
        if padding is None:
            assert opcode in XC_BCONV2D_OPCODES
        elif padding is not Padding.VALID:
            raise ValueError(f"Unsupported padding: {padding}")
        output_size = calculate_valid_output_size(
            input_size, strides, weight_shape[1:3]
        )

    tout = subgraph.create_tensor(
        "output",
        output_tensor_type,
        shape=(1, *output_size, C_out),
        isoutput=True,
        quantization=output_quantization,
    )

    # create custom options
    custom_options = {"padding": padding} if padding else {}
    if opcode is ExternalOpCodes.LceBconv2d:
        custom_options.update(
            {
                "channels_in": C_in,
                "fused_activation_function": ActivationFunctionType.NONE,
                "stride_height": strides[0],
                "stride_width": strides[1],
                "dilation_width_factor": 1,
                "dilation_height_factor": 1,
            }
        )
    else:
        custom_options["stride"] = strides

    # create operator
    subgraph.create_operator(
        OperatorCode(opcode),
        inputs=input_tensors,
        outputs=[tout],
        custom_options=custom_options,
    )

    return subgraph.model


def build_lceBconv2d(
    subgraph: Optional[Subgraph] = None, *, padding: Padding, **kwargs
) -> XCOREModel:
    return build_bconv2d(
        subgraph, padding=padding, opcode=ExternalOpCodes.LceBconv2d, **kwargs,
    )


def build_XC_bconv2d(
    subgraph: Optional[Subgraph] = None,
    *,
    opcode: XCOREOpCodes = XCOREOpCodes.XC_bconv2d_int8,
    padding: Optional[Padding] = None,
    **kwargs,
) -> XCOREModel:
    return build_bconv2d(subgraph, padding=padding, opcode=opcode, **kwargs)


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_mutate(
    trf_pass: ModelTransformationPass, model: XCOREModel, new_opcode: XCOREOpCodes
) -> None:
    subgraph = model.subgraphs[0]
    assert len(subgraph.operators) == 1

    _test_mutate(trf_pass, model, new_opcode)


def test_bconv2d_mutate(
    trf_pass: ReplaceBconv2DPass, model: XCOREModel, new_opcode: XCOREOpCodes
) -> None:
    subgraph = model.subgraphs[0]
    operators = subgraph.operators
    op = operators[-1]
    strides = op.custom_options["stride_height"], op.custom_options["stride_width"]
    padding = op.custom_options["padding"]

    test_conv2d_mutate(trf_pass, model, new_opcode)

    assert len(operators) == 1

    new_op = operators[-1]
    assert "illegal_params" in new_op.custom_options
    assert "stride" in new_op.custom_options
    assert strides == new_op.custom_options["stride"]
    assert "padding" in new_op.custom_options
    assert padding == new_op.custom_options["padding"]
