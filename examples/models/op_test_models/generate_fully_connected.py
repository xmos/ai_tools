#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
from pathlib import Path
import numpy as np
from tflite2xcore.model_generation import utils
import tensorflow as tf
import op_test_models_common as common

DEFAULT_OUTPUT_DIM = 10
DEFAULT_INPUT_DIM = 32
DEFAULT_EPOCHS = 5 * (DEFAULT_OUTPUT_DIM - 1)
DEFAULT_BS = 128
DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'fully_connected').resolve()


def main(raw_args=None):
    parser = common.OpTestFCParser(defaults={
        'path': DEFAULT_PATH,
        'input_dim': DEFAULT_INPUT_DIM,
        'output_dim': DEFAULT_OUTPUT_DIM,
        'batch_size': DEFAULT_BS,
        'epochs': DEFAULT_EPOCHS,
        'inits': {
            'weight_init': {
                'type': common.OpTestInitializers.UNIF,
                'help': "Initializer for weight distribution."
            },
            'bias_init': {
                'type': common.OpTestInitializers.CONST,
                'help': "Initializer for bias distribution."
            }
        }
    })
    args = parser.parse_args(raw_args)
    utils.set_gpu_usage(args.use_gpu, args.verbose)

    model = common.OpTestDefaultFCModel('fc_deepin_deepout', args.path)
    model.run(train_model=args.train_model,
              input_dim=args.input_dim,
              output_dim=args.output_dim,
              batch_size=args.batch_size,
              epochs=args.epochs,
              **args.inits)


if __name__ == "__main__":
    main()
