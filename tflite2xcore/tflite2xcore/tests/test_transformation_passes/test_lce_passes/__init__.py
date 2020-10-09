# Copyright (c) 2020, XMOS Ltd, All rights reserved

import numpy as np
from copy import deepcopy
from typing import Tuple, Optional

from tflite2xcore.transformation_passes import ModelTransformationPass
from tflite2xcore.xcore_model import XCOREModel, Subgraph
from tflite2xcore.xcore_schema import (
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
    generate_dummy_int8_data,
    generate_dummy_int32_data,
)

from ..conftest import ParamsType
from ..conftest import (  # pylint: disable=unused-import
    _make_name_type_pairs,
    _test_non_matching_params,
    test_matching_params,
    test_non_matching_tensors,
    test_replace_mutate as _test_mutate,
)
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
    {"input_channels": [256, 512, 1024], "non_matching_input_channels": [48, 128, 245]}
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
    subgraph = subgraph or XCOREModel().create_subgraph()
    height, width, channels = input_shape
    input_shape = (1, height, width, channels)
    output_shape = (1, height, width, int(np.ceil(channels / 32)))

    tin = subgraph.create_tensor("input", input_tensor_type, input_shape, isinput=True)
    tout = subgraph.create_tensor(
        "output", TensorType.INT32, output_shape, isoutput=True,
    )

    subgraph.create_operator(
        OperatorCode(ExternalOpCodes.add_new_opcode("LceQuantize")),
        inputs=[tin],
        outputs=[tout],
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
    subgraph = subgraph or XCOREModel().create_subgraph()

    # the given shapes are not bitpacked (i.e. true channel counts)
    # so we bitpack them
    C_out, _, _, C_in = weight_shape
    bitpacked_input_channels = int(np.ceil(C_in / 32))
    weight_shape = (*weight_shape[:3], bitpacked_input_channels)

    input_shape = [1, *input_size, bitpacked_input_channels]
    tin = subgraph.create_tensor("input", TensorType.INT32, input_shape, isinput=True)
    w = subgraph.create_tensor("weights", TensorType.INT32, weight_shape)
    output_threshold = subgraph.create_tensor(
        "output_threshold", TensorType.INT32, weight_shape[:1]
    )

    # add dummy data so that the op can be mutated
    w.buffer.data = generate_dummy_int8_data(w.shape)
    output_threshold.buffer.data = generate_dummy_int32_data(output_threshold.shape)

    # check padding and determine output size
    if padding is Padding.SAME:
        output_size = calculate_same_output_size(input_size, strides)
    else:
        if padding is None:
            assert opcode in (
                XCOREOpCodes.XC_bconv2d_bin_out,
                XCOREOpCodes.XC_bconv2d_int8_out,
            )
        elif padding is not Padding.VALID:
            raise ValueError(f"Unsupported padding: {padding}")
        output_size = calculate_valid_output_size(
            input_size, strides, weight_shape[1:3]
        )

    tout = subgraph.create_tensor(
        "output", output_tensor_type, shape=(1, *output_size, C_out), isoutput=True
    )

    # create custom options
    custom_options = {"padding": padding} if padding else {}
    if opcode is ExternalOpCodes.add_new_opcode("LceBconv2d"):
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
        inputs=[tin, w, output_threshold],
        outputs=[tout],
        custom_options=custom_options,
    )

    return subgraph.model


def build_lceBconv2d(
    subgraph: Optional[Subgraph] = None, *, padding: Padding, **kwargs
) -> XCOREModel:
    return build_bconv2d(
        subgraph,
        padding=padding,
        opcode=ExternalOpCodes.add_new_opcode("LceBconv2d"),
        **kwargs,
    )


def build_XC_bconv2d(
    subgraph: Optional[Subgraph] = None,
    *,
    opcode: XCOREOpCodes = XCOREOpCodes.XC_bconv2d_int8_out,
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
