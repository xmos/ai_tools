#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
from pathlib import Path
import numpy as np
from tflite2xcore.model_generation import utils
from tflite2xcore.model_generation.interface import KerasModel
import tensorflow as tf
import op_test_models_common as common

DEFAULT_OUTPUT_DIM = 10
DEFAULT_INPUT_DIM = 32
DEFAULT_EPOCHS = 5 * (DEFAULT_OUTPUT_DIM - 1)
DEFAULT_BS = 128
DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'fully_connected').resolve()


def main(path=DEFAULT_PATH, *,
         input_dim=DEFAULT_INPUT_DIM,
         output_dim=DEFAULT_OUTPUT_DIM,
         train_new_model=False,
         batch_size=DEFAULT_BS,
         epochs=DEFAULT_EPOCHS,
         inits=common.DEFAULT_INITS_WB):
    kwargs = {
        'name': 'fc_deepin_anyout',
        'path': path if path else DEFAULT_PATH
    }
    model = common.DefaultOpTestFCModel(**kwargs)
    model.run(train_new_model=train_new_model,
              input_dim=input_dim,
              output_dim=output_dim,
              inits=inits,
              batch_size=batch_size,
              epochs=epochs)


if __name__ == "__main__":
    parser = common.OpTestFCParser(defaults={
        'path': DEFAULT_PATH,
        'input_dim': DEFAULT_INPUT_DIM,
        'output_dim': DEFAULT_OUTPUT_DIM,
        'batch_size': DEFAULT_BS,
        'epochs': DEFAULT_EPOCHS
    })
    args = parser.parse_args()
    utils.set_verbosity(args.verbose)
    utils.set_gpu_usage(args.use_gpu, args.verbose)

    main(path=args.path,
         input_dim=args.input_dim,
         output_dim=args.output_dim,
         train_new_model=args.train_model,
         batch_size=args.batch_size,
         epochs=args.epochs,
         inits=args.inits)
