from typing import Optional
from enum import Enum, unique
from sys import stderr
from tensorflow.keras import Model, layers

from .Conv2DOptimisations import Conv2DOptimisation
from .Strictness import Strictness
from .XCoreUnoptimisedError import XCoreUnoptimisedError


# Format and print warning
def print_warning(layer_idx: int, layer_name: str, message: str):
    print(f"Layer {layer_idx} ({layer_name}): {message}", file=stderr)


# Run validation on individual layer
def __validate_conv2d_layer__(
        layer: layers.Conv2D, idx: int, strictness: Strictness, allow_padding: Optional[bool] = False
) -> None:
    num_filters: int = layer.filters
    channel_idx: int = 1 if layer.data_format == "channels_first" else 3
    num_input_channels: int = layer.input.shape[channel_idx]
    is_padded: bool = layer.padding == "same"
    current_optimisation: Conv2DOptimisation

    # Determine what optimisation will be made
    if num_filters % 4 != 0 or num_input_channels % 4 != 0:
        current_optimisation = Conv2DOptimisation.DEFAULT
    elif is_padded:
        current_optimisation = Conv2DOptimisation.PADDED_INDIRECT
    elif (num_input_channels % 32 == 0) and (num_filters % 16 == 0):
        current_optimisation = Conv2DOptimisation.VALID_DIRECT
    else:
        current_optimisation = Conv2DOptimisation.VALID_INDIRECT

    if current_optimisation == Conv2DOptimisation.DEFAULT:
        message: str = "Output and input depth must be a multiple of four. Reference design will be used."
        if strictness == Strictness.ERROR:
            raise XCoreUnoptimisedError(message, idx)
        print_warning(layer_idx=idx, layer_name=layer.name, message=message)
    elif current_optimisation == Conv2DOptimisation.PADDED_INDIRECT and not allow_padding:
        message: str = f"Padding comes with a a significant performance hit. Consider whether it is truly needed."
        if strictness == Strictness.ERROR:
            raise XCoreUnoptimisedError(message, idx)
        print_warning(layer_idx=idx, layer_name=layer.name, message=message)
    elif current_optimisation == Conv2DOptimisation.VALID_INDIRECT:
        message: str = f"Input depth is not a multiple of 32 or output depth is not a multiple of 16." "Valid Indirect will be used "
        if strictness == Strictness.ERROR:
            raise XCoreUnoptimisedError(message, idx)
        print_warning(layer_idx=idx, layer_name=layer.name, message=message)


# Takes in a keras model and analyses it to make sure that any Conv2Ds can be optimised by xformer as much as possible.
def validate_conv2d(
        model: Model, strictness: Strictness, allow_padding: Optional[bool] = False
) -> None:
    for (idx, layer) in enumerate(model.layers):
        if type(layer) != layers.Conv2D:
            continue
        __validate_conv2d_layer__(layer, idx, strictness, allow_padding)


# Run validation on individual layer
def __validate_depthwise_conv2d_layer__(
        layer: layers.Conv2D, idx: int, strictness: Strictness, allow_padding: Optional[bool] = False
):
    num_filters: int = layer.filters
    channel_idx: int = 1 if layer.data_format == "channels_first" else 3
    num_input_channels: int = layer.input.shape[channel_idx]
    is_padded: bool = layer.padding == "same"
    current_optimisation: Conv2DOptimisation

    # Determine what optimisation will be made
    if num_filters % 4 != 0 or num_input_channels % 4 != 0:
        current_optimisation = Conv2DOptimisation.DEFAULT
    elif is_padded:
        current_optimisation = Conv2DOptimisation.PADDED_INDIRECT
    else:
        current_optimisation = Conv2DOptimisation.VALID_DIRECT

    # Raise Errors or Warnings
    if current_optimisation == Conv2DOptimisation.DEFAULT:
        message: str = "Output and input depth must be a multiple of four. Reference design will be used."
        if strictness == Strictness.ERROR:
            raise XCoreUnoptimisedError(message, idx)
        print_warning(message=message, layer_name=layer.name, layer_idx=idx)
    elif current_optimisation == Conv2DOptimisation.PADDED_INDIRECT and not allow_padding:
        message: str = f"Padding comes with a a significant performance hit. Consider whether it is truly needed."
        if strictness == Strictness.ERROR:
            raise XCoreUnoptimisedError(message, idx)
        print_warning(message=message, layer_name=layer.name, layer_idx=idx)


# Takes in a keras model and analyses it to make sure that any Conv2Ds can be optimised by xformer as much as possible.
def validate_depthwise_conv2d(
        model: Model, strictness: Strictness, allow_padding: Optional[bool] = False
) -> None:
    for (idx, layer) in enumerate(model.layers):
        if type(layer) != layers.DepthwiseConv2D:
            continue
        __validate_depthwise_conv2d_layer__(layer, idx, strictness, allow_padding)


# Validate all layers
def validate(model: Model, strictness: Strictness, allow_padding: Optional[bool] = False) -> None:
    for (idx, layer) in enumerate(model.layers):
        if type(layer) == layers.DepthwiseConv2D:
            __validate_depthwise_conv2d_layer__(layer, idx, strictness, allow_padding)
        elif type(layer) == layers.Conv2D:
            __validate_conv2d_layer__(layer, idx, strictness, allow_padding)
    return
