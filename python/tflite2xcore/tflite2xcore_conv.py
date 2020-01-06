#!/usr/bin/env python
#
# Copyright (c) 2019, XMOS Ltd, All rights reserved

import os
import logging
import argparse

from tflite2xcore.graph_transformer import PassManager, PassPriority
from tflite2xcore.tflite_utils import DEFAULT_FLATC, DEFAULT_SCHEMA, set_gpu_usage
from tflite2xcore import read_flatbuffer, write_flatbuffer
from tflite2xcore import transformation_passes as passes


def main(tflite_input_path, tflite_output_path, *,
         is_classifier=False, remove_softmax=False):
    model = read_flatbuffer(tflite_input_path)
    pass_mgr = PassManager(
        model,
        passes=[
            passes.RemoveQuantizerFloatInputPass(),
            passes.RemoveDequantizerFloatOutputPass()
        ]
    )

    if is_classifier or remove_softmax:
        pass_mgr.register_pass(passes.RemoveSoftmaxOutputPass())

    pass_mgr.register_pass(passes.ReplaceDeepinDeepoutConv2DPass())
    pass_mgr.register_pass(passes.ReplaceShallowinDeepoutConv2DPass())
    pass_mgr.register_pass(passes.ReplaceSingleinDeepoutDepthwiseConv2DPass())
    pass_mgr.register_pass(passes.ReplaceDeepMaxpool2DPass())
    pass_mgr.register_pass(passes.ReplaceDeepinShallowoutFullyConnectedOutputPass())
    pass_mgr.register_pass(passes.RemoveUnusedBuffersPass())

    pass_mgr.run_passes()

    model.description = 'TOCO + XMOS converted.'

    write_flatbuffer(model, tflite_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('tflite_input', help='Input .tflite file.')
    parser.add_argument('tflite_output', help='Output .tflite file.')
    parser.add_argument('--classifier', action='store_true', default=False,
                        help="Apply optimizations for classifier networks "
                             "(e.g. softmax removal and output argmax).")
    parser.add_argument('--remove_softmax', action='store_true', default=False,
                        help="Remove output softmax operation.")
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Verbose mode.')
    args = parser.parse_args()

    verbose = args.verbose

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.getLogger('tensorflow').setLevel(logging.ERROR)

    set_gpu_usage(False, verbose)

    tflite_input_path = os.path.realpath(args.tflite_input)
    tflite_output_path = os.path.realpath(args.tflite_output)
    is_classifier = args.classifier
    remove_softmax = args.remove_softmax

    main(tflite_input_path, tflite_output_path,
         is_classifier=is_classifier, remove_softmax=remove_softmax)
