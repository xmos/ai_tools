#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
from abc import abstractmethod
import argparse
from pathlib import Path
from tflite2xcore.model_generation import utils
from tflite2xcore.model_generation.interface import KerasModel
import tensorflow as tf

DEFAULT_INPUTS = 36
DEFAULT_HEIGHT = 5
DEFAULT_WIDTH = 9
DEFAULT_POOL_HEIGHT = 3
DEFAULT_POOL_WIDTH = 5
DEFAULT_POOL_SIZE = (DEFAULT_POOL_HEIGHT, DEFAULT_POOL_WIDTH)
DEFAULT_PADDING = 'valid'
DEFAULT_STRIDE_HEIGHT = 1
DEFAULT_STRIDE_WIDTH = 2
DEFAULT_STRIDES = (DEFAULT_STRIDE_HEIGHT, DEFAULT_STRIDE_WIDTH)
DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'avgpool2d').resolve()


# TODO: refactor this since other single op models also use something similar
class DefaultAvgPool2DModel(KerasModel):
    @abstractmethod
    def build_core_model(self, *args, **kwargs):
        pass

    def build(self, *args, **kwargs):
        self._prep_backend()
        self.build_core_model(*args, **kwargs)
        self.core_model.compile(optimizer='adam',
                                loss='sparse_categorical_crossentropy',
                                metrics=['accuracy'])
        self.core_model.summary()

    def train(self):  # Not training this model
        pass

    def prep_data(self):  # Not training this model
        pass

    def gen_test_data(self):
        self.data['export_data'], self.data['quant'] = utils.generate_dummy_data(*self.input_shape)

    def run(self):
        self.gen_test_data()
        self.save_core_model()
        self.populate_converters()


class AvgPool2D(DefaultAvgPool2DModel):
    def build_core_model(self, height, width, input_channels,
                         *, pool_size, strides, padding):
        assert input_channels % 4 == 0, "# of input channels must be multiple of 4"
        assert padding.lower() == 'valid', "padding mode must be valid"
        assert (height % 2 == 1 or width % 2 == 1
                or pool_size[0] != 2 or pool_size[1] != 2
                or strides[0] != 2 or strides[1] != 2), "parameters must differ from what avgpool2d_2x2 can match"

        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.layers.AveragePooling2D(
                    pool_size=pool_size,
                    strides=strides,
                    padding=padding,
                    input_shape=(height, width, input_channels))
            ]
        )


def main(path=DEFAULT_PATH, *,
         input_channels=DEFAULT_INPUTS,
         height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH,
         pool_size=DEFAULT_POOL_SIZE,
         strides=DEFAULT_STRIDES,
         padding=DEFAULT_PADDING):
    model = AvgPool2D('avgpool2d', Path(path))
    model.build(height, width, input_channels,
                pool_size=pool_size,
                strides=strides,
                padding=padding)
    model.run()


class DefaultAvgPool2DParser(argparse.ArgumentParser):
    def __init__(self, *args, defaults, **kwargs):
        kwargs['formatter_class'] = argparse.ArgumentDefaultsHelpFormatter
        super().__init__(*args, **kwargs)
        self.add_argument(
            'path', nargs='?', default=defaults['path'],
            help='Path to a directory where models and data will be saved in subdirectories.')
        self.add_argument(
            '-in', '--inputs', type=int, default=defaults['inputs'],
            help='Number of input channels')
        self.add_argument(
            '-hi', '--height', type=int, default=defaults['height'],
            help='Height of input image')
        self.add_argument(
            '-wi', '--width', type=int, default=defaults['width'],
            help='Width of input image')
        self.add_argument(
            '-v', '--verbose', action='store_true', default=False,
            help='Verbose mode.')


def strides_pool_arg_handler(args):
    parameters = {
        'strides': (DEFAULT_STRIDE_HEIGHT, DEFAULT_STRIDE_WIDTH),
        'pool_size': (DEFAULT_POOL_HEIGHT, DEFAULT_POOL_WIDTH)
    }
    arguments = {k: vars(args)[k] for k in parameters if k in vars(args)}
    for k in arguments:
        params = arguments[k]
        if len(params) > 2:
            raise argparse.ArgumentTypeError(
                f"The {k} argument must be at most 2 numbers.")
        else:
            arguments[k] = tuple(params) if len(params) == 2 else (params[0]) * 2

    return arguments


if __name__ == "__main__":
    parser = DefaultAvgPool2DParser(defaults={
        'path': DEFAULT_PATH,
        'inputs': DEFAULT_INPUTS,
        'height': DEFAULT_HEIGHT,
        'width': DEFAULT_WIDTH
    })
    parser.add_argument(
        '-st', '--strides', nargs='+', type=int, default=argparse.SUPPRESS,
        help="Strides, vertical first "
             f"(default: {DEFAULT_STRIDE_HEIGHT} {DEFAULT_STRIDE_WIDTH})")
    parser.add_argument(
        '-po', '--pool_size', nargs='*', type=int, default=argparse.SUPPRESS,
        help="Pool size:, vertical first "
             f"(default: {DEFAULT_POOL_HEIGHT} {DEFAULT_POOL_WIDTH})")
    parser.add_argument(
        '-pd', '--padding', type=str, default=DEFAULT_PADDING,
        help='Padding mode')
    args = parser.parse_args()

    utils.set_verbosity(args.verbose)
    utils.set_gpu_usage(False, args.verbose)

    strides_pool = strides_pool_arg_handler(args)

    main(path=args.path,
         input_channels=args.inputs,
         height=args.height, width=args.width,
         padding=args.padding,
         **strides_pool
         )
