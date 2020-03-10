# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from ..conftest import (
    PARAMS,
    _test_non_matching_dim,
    test_matching_params,
    test_non_matching_output_channels,
    test_non_matching_kernel_height,
    test_non_matching_kernel_width,
    test_non_matching_input_channels,
    test_non_matching_types,
    test_non_matching_stride_h,
    test_non_matching_stride_w
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PARAMS["default"].update({
    "output_channels": [16, 32],
    "non_matching_output_channels": [8, 24, 17, 63],
    "kernel_height": [1, 3, 5, 7],  # TODO: the parity constraint is deprecated after the conv2d improvements
    "non_matching_kernel_height": [2, 4, 6],  # TODO: the parity constraint is deprecated after the conv2d improvements
    "kernel_width": [1, 3, 5, 7]  # TODO: the parity constraint is deprecated after the conv2d improvements
})

PARAMS["smoke"].update({
    "output_channels": [16],
    "non_matching_output_channels": [8],
    "kernel_height": [1, 3],  # TODO: the parity constraint is deprecated after the conv2d improvements
    "non_matching_kernel_height": [2],  # TODO: the parity constraint is deprecated after the conv2d improvements
    "kernel_width": [1, 3]  # TODO: the parity constraint is deprecated after the conv2d improvements
})
