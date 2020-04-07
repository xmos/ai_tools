#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

from pathlib import Path
import op_test_models_common as common

DEFAULT_INPUTS = 20
DEFAULT_OUTPUTS = 20
DEFAULT_HEIGHT = 5
DEFAULT_WIDTH = DEFAULT_HEIGHT
DEFAULT_KERNEL_HEIGHT = 3
DEFAULT_KERNEL_WIDTH = DEFAULT_KERNEL_HEIGHT
DEFAULT_PADDING = 'same'
DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'conv2d_deep').resolve()
DEFAULT_NUM_THREADS = 1


class Conv2DDeep(common.OpTestDefaultConvModel):
    def build_core_model(self, *args, **kwargs):
        K_h, K_w, _, _, input_channels, output_channels = args
        assert output_channels % 4 == 0, "# of output channels must be multiple of 4"
        assert input_channels % 4 == 0, "# of input channels must be multiple of 4"
        assert K_h != 1 or K_w != 1, "1x1 kernel is not allowed for DIDO testing"
        super().build_core_model(*args, **kwargs)


def main(raw_args=None):
    parser = common.OpTestConvParser(defaults={
        'path': DEFAULT_PATH,
        'inputs': DEFAULT_INPUTS,
        'outputs': DEFAULT_OUTPUTS,
        'width': DEFAULT_WIDTH,
        'height': DEFAULT_HEIGHT,
        'padding': DEFAULT_PADDING,
        'kernel_width': DEFAULT_KERNEL_WIDTH,
        'kernel_height': DEFAULT_KERNEL_HEIGHT,
        'inits': {
            'input_init': {'type': common.OpTestInitializers.UNIF},
            'weight_init': {'type': common.OpTestInitializers.UNIF},
            'bias_init': {'type': common.OpTestInitializers.CONST}
        }
    })
    parser.add_argument(  # TODO: use the a better parser for this after the conv2d enhancements
        "-st", "--strides", nargs=2, type=int, default=[1, 1],
        help=f"Strides, vertical first",
    )
    parser.add_argument(
        '-par', '--num_threads', type=int, default=DEFAULT_NUM_THREADS,
        help='Number of parallel threads for xcore.ai optimization.')
    args = parser.parse_args(raw_args)
    args.strides = tuple(args.strides)  # TODO: fix this

    model = Conv2DDeep('conv2d_deep', args.path)
    model.build(args.kernel_height, args.kernel_width,
                args.height, args.width,
                args.inputs, args.outputs,
                padding=args.padding,
                strides=args.strides,
                **args.inits)
    model.run(num_threads=args.num_threads)


if __name__ == "__main__":
    main()
