# Copyright (c) 2020, XMOS Ltd, All rights reserved

import argparse

from pathlib import Path

from tflite2xcore.model_generation import utils


class DefaultParser(argparse.ArgumentParser):
    def __init__(self, *args, defaults, **kwargs):
        kwargs["formatter_class"] = argparse.ArgumentDefaultsHelpFormatter
        super().__init__(*args, **kwargs)
        self.add_argument(
            "-path", nargs="?", default=defaults["path"],
            help="Path to a directory where models and data will be saved in subdirectories.",
        )
        self.add_argument(
            "-v", "--verbose", action="store_true", default=False,
            help="Verbose mode."
        )

    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)
        utils.set_verbosity(args.verbose)
        utils.set_gpu_usage(args.use_gpu if hasattr(args, 'use_gpu') else False,
                            args.verbose)
        args.path = Path(args.path)
        return args


class InitializerParser(DefaultParser):
    def __init__(self, *args, defaults, **kwargs):
        super().__init__(*args, defaults=defaults, **kwargs)

        self.seed = None
        self.add_argument(
            "--seed", type=int,
            help="Set the seed value for the initializers."
        )

        self._default_handler(defaults)

    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)
        self.seed = args.seed
        self._initializer_args_handler(args)
        return args

    def _default_handler(self, defaults):
        pass

    def _initializer_args_handler(self, args):
        pass


class TrainableParser(InitializerParser):
    def __init__(self, *args, defaults, **kwargs):
        super().__init__(*args, defaults=defaults, **kwargs)
        self.add_argument(
            "--train_model", action="store_true", default=False,
            help="Train new model instead of loading pretrained tf.keras model.",
        )
        self.add_argument(
            "--use_gpu", action="store_true", default=False,
            help="Use GPU for training. Might result in non-reproducible results.",
        )
        self.add_argument(
            "-bs", "--batch_size", type=int, default=defaults["batch_size"],
            help="Set the training batch size."
        )
        self.add_argument(
            "-ep", "--epochs", type=int, default=defaults["epochs"],
            help="Set the number of training epochs size."
        )
