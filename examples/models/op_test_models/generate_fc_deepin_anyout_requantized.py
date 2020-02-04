#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import argparse
from pathlib import Path
from tflite2xcore.model_generation import utils
import tflite2xcore.converter as xcore_conv
from tflite2xcore import read_flatbuffer, write_flatbuffer

from generate_fc_deepin_anyout import (
    DEFAULT_OUTPUT_DIM, DEFAULT_INPUT_DIM, DEFAULT_EPOCHS, DEFAULT_BS
)
DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'fc_deepin_anyout_requantized').resolve()

from generate_fc_deepin_anyout import FcDeepinAnyout, run_main


class FcDeepinAnyoutRequantized(FcDeepinAnyout):
    def to_tf_xcore(self):
        assert 'model_quant' in self.models
        self.models['model_xcore'] = str(self.models['models_dir'] / 'model_xcore.tflite')
        model = read_flatbuffer(str(self.models['model_quant']))

        # NOTE: since the output softmax is not removed during the first
        # conversion, ReplaceFullyConnectedIntermediatePass will
        # match and insert the requantization. Then, the softmax can be removed.
        xcore_conv.optimize_for_xcore(model, is_classifier=False, remove_softmax=False)
        xcore_conv.strip_model(model, remove_softmax=True)

        model = write_flatbuffer(model, self.models['model_xcore'])


def main(path=DEFAULT_PATH, *,
         input_dim=DEFAULT_INPUT_DIM, output_dim=DEFAULT_OUTPUT_DIM,
         train_new_model=False):
    run_main(FcDeepinAnyoutRequantized('fc_deepin_anyout_requantized', Path(path)),
             train_new_model=train_new_model, input_dim=input_dim, output_dim=output_dim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'path', nargs='?', default=DEFAULT_PATH,
        help='Path to a directory where models and data will be saved in subdirectories.')
    parser.add_argument(
        '--use_gpu', action='store_true', default=False,
        help='Use GPU for training. Might result in non-reproducible results.')
    parser.add_argument(
        '-out', '--output_dim', type=int, default=DEFAULT_OUTPUT_DIM,
        help='Number of output dimensions, must be at least 2.')
    parser.add_argument(
        '-in', '--input_dim', type=int, default=DEFAULT_INPUT_DIM,
        help='Input dimension.')
    parser.add_argument(
        '--train_model', action='store_true', default=False,
        help='Train new model instead of loading pretrained tf.keras model.')
    parser.add_argument(
        '-v', '--verbose', action='store_true', default=False,
        help='Verbose mode.')
    args = parser.parse_args()

    utils.set_verbosity(args.verbose)
    utils.set_gpu_usage(args.use_gpu, args.verbose)

    main(path=args.path,
         input_dim=args.input_dim, output_dim=args.output_dim,
         train_new_model=args.train_model)
