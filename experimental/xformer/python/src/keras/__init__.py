from enum import Enum, unique
from sys import stderr
from tensorflow.keras import Model, layers


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


@unique
class Strictness(Enum):
    # Throw XCoreUnoptimisedError if not optimised
    ERROR = 0

    # Print warnings to stderr
    WARNING = 1


@unique
class ConvOptimisation(Enum):
    DEFAULT = 0
    PADDED_INDIRECT = 1
    VALID_INDIRECT = 2
    VALID_DIRECT = 3


class XCoreUnoptimisedError(Exception):
    __layer_idx__: int

    def __init__(self, message: str, layer_idx: int):
        self.__layer_idx__ = layer_idx
        super().__init__(message)

    def get_layer_idx(self) -> int:
        return self.__layer_idx__


"""
Takes in a keras model and analyses it to make sure that any Conv2Ds can be optimised by xformer as much as possible.
"""


def validate_conv2d(
    model: Model, strictness: Strictness, allow_padding: bool = False
) -> None:
    for (idx, layer) in enumerate(model.layers):
        if type(layer) != layers.Conv2D:
            continue

        num_filters: int = layer.filters
        channel_idx: int = 1 if layer.data_format == "channels_first" else 3
        num_input_channels: int = layer.input.shape[channel_idx]
        is_padded: bool = layer.padding == "same"
        current_optimisation: ConvOptimisation

        # Determine what optimisation will be made
        if num_filters % 4 != 0 or num_input_channels % 4 != 0:
            current_optimisation = ConvOptimisation.DEFAULT
        elif is_padded:
            current_optimisation = ConvOptimisation.PADDED_INDIRECT
        elif (num_input_channels % 32 == 0) and (num_filters % 16 == 0):
            current_optimisation = ConvOptimisation.VALID_DIRECT
        else:
            current_optimisation = ConvOptimisation.VALID_INDIRECT

        def print_warning():
            print(f"Layer {idx} ({layer.name}): {message}", file=stderr)

        if current_optimisation == ConvOptimisation.DEFAULT:
            message: str = "Output and input depth must be a multiple of four. Reference design will be used."
            if strictness == Strictness.ERROR:
                raise XCoreUnoptimisedError(message, idx)
            print_warning()
        elif (
            current_optimisation == ConvOptimisation.PADDED_INDIRECT
            and not allow_padding
        ):
            message: str = f"Padding comes with a a significant performance hit. Consider whether it is truly needed."
            if strictness == Strictness.ERROR:
                raise XCoreUnoptimisedError(message, idx)
            print_warning()
        elif current_optimisation == ConvOptimisation.VALID_INDIRECT:
            message: str = f"Input depth is not a multiple of 32 or output depth is not a multiple of 16. Valid Indirect will be used "
            if strictness == Strictness.ERROR:
                raise XCoreUnoptimisedError(message, idx)
            print_warning()


"""
Takes in a keras model and analyses it to make sure that any Conv2Ds can be optimised by xformer as much as possible.
"""


def validate_depthwise_conv2d(
    model: Model, strictness: Strictness, allow_padding: bool = False
) -> None:
    for (idx, layer) in enumerate(model.layers):
        if type(layer) != layers.DepthwiseConv2D:
            continue

        num_filters: int = layer.filters
        channel_idx: int = 1 if layer.data_format == "channels_first" else 3
        num_input_channels: int = layer.input.shape[channel_idx]
        is_padded: bool = layer.padding == "same"
        current_optimisation: ConvOptimisation

        # Determine what optimisation will be made
        if num_filters % 4 != 0 or num_input_channels % 4 != 0:
            current_optimisation = ConvOptimisation.DEFAULT
        elif is_padded:
            current_optimisation = ConvOptimisation.PADDED_INDIRECT
        else:
            current_optimisation = ConvOptimisation.VALID_DIRECT

        def print_warning():
            print(f"Layer {idx} ({layer.name}): {message}", file=stderr)

        if current_optimisation == ConvOptimisation.DEFAULT:
            message: str = "Output and input depth must be a multiple of four. Reference design will be used."
            if strictness == Strictness.ERROR:
                raise XCoreUnoptimisedError(message, idx)
            print_warning()
        elif (
            current_optimisation == ConvOptimisation.PADDED_INDIRECT
            and not allow_padding
        ):
            message: str = f"Padding comes with a a significant performance hit. Consider whether it is truly needed."
            if strictness == Strictness.ERROR:
                raise XCoreUnoptimisedError(message, idx)
            print_warning()


def validate(model: Model, strictness: Strictness, allow_padding: bool = False) -> None:
    validate_conv2d(model, strictness, allow_padding)
    return
