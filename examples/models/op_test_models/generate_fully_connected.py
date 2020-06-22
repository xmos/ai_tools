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
DEFAULT_NUM_THREADS = 1
DEFAULT_PATH = Path(__file__).parent.joinpath("debug", "fully_connected").resolve()


def main(raw_args=None):
    parser = common.OpTestFCParser(
        defaults={
            "path": DEFAULT_PATH,
            "input_dim": DEFAULT_INPUT_DIM,
            "output_dim": DEFAULT_OUTPUT_DIM,
            "batch_size": DEFAULT_BS,
            "epochs": DEFAULT_EPOCHS,
            "inits": {
                "weight_init": {"type": common.OpTestInitializers.UNIF},
                "bias_init": {"type": common.OpTestInitializers.CONST},
            },
        }
    )
    parser.add_argument(
        "-par",
        "--num_threads",
        type=int,
        default=DEFAULT_NUM_THREADS,
        help="Number of parallel threads for xcore.ai optimization.",
    )
    args = parser.parse_args(raw_args)

    model = common.OpTestDefaultFCModel("fully_connected", args.path)
    if args.train_model:
        model.build_and_train(
            args.input_dim, args.output_dim, args.batch_size, args.epochs, **args.inits
        )
    else:
        model.load_core_model(args.output_dim)
    model.convert_and_save(xcore_num_threads=args.num_threads)


if __name__ == "__main__":
    main()
