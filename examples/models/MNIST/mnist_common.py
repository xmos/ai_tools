# Copyright (c) 2020, XMOS Ltd, All rights reserved

import logging
import argparse
from pathlib import Path
from tflite2xcore.model_generation import utils
from tflite2xcore.model_generation.interface import KerasClassifier


class MNISTModel(KerasClassifier):
    def __init__(self, *args, use_aug=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_aug = use_aug

    def prep_data(self, *, simard_resize=False, padding=2):
        self.data = utils.prepare_MNIST(self._use_aug, simard=simard_resize, padding=padding)
        for k, v in self.data.items():
            logging.debug(f"Prepped data[{k}] with shape: {v.shape}")

    def gen_test_data(self):
        if not self.data:
            self.prep_data()
        self.data['export_data'] = self.data['x_test'][:10]
        self.data['quant'] = self.data['x_train'][:10]

    def run(self, train_new_model=False, epochs=None, batch_size=None):
        if train_new_model:
            assert epochs
            assert batch_size
            # Build model and compile
            self.build()
            # Prepare training data
            self.prep_data()
            # Train model
            self.train(batch_size=batch_size, epochs=epochs)
            self.save_core_model()
        else:
            # Recover previous state from file system
            self.load_core_model()
        # Generate test data
        self.gen_test_data()
        # Populate converters
        self.populate_converters()


#  ----------------------------------------------------------------------------
#                                   PARSERS
#  ----------------------------------------------------------------------------


class CommonDefaultParser(argparse.ArgumentParser):

    def __init__(self, *args, defaults, **kwargs):
        kwargs["formatter_class"] = argparse.ArgumentDefaultsHelpFormatter
        super().__init__(*args, **kwargs)
        self.add_argument(
            'path', nargs='?', default=defaults['path'],
            help='Path to a directory where models and data will be saved in subdirectories.')
        self.add_argument(
            '--name', type=str, default=defaults['name'],
            help='Name of the model, used in the creation of the model itself and subdirectories.')
        self.add_argument(
            '--use_gpu', action='store_true', default=False,
            help='Use GPU for training. Might result in non-reproducible results.')
        self.add_argument(
            '--train_model', action='store_true', default=False,
            help='Train new model instead of loading pretrained tf.keras model.')
        self.add_argument(
            '--classifier', action='store_true', default=False,
            help='Apply classifier optimizations during xcore conversion.')
        self.add_argument(
            '-bs', '--batch', type=int, default=defaults['batch_size'],
            help='Batch size.')
        self.add_argument(
            '-ep', '--epochs', type=int, default=defaults['epochs'],
            help='Number of epochs.')
        self.add_argument(
            '-aug', '--augment_dataset', action='store_true', default=False,
            help='Create a dataset with elastic transformations.')
        self.add_argument(
            '-v', '--verbose', action='store_true', default=False,
            help='Verbose mode.')

    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)
        utils.set_verbosity(args.verbose)
        utils.set_gpu_usage(args.use_gpu, args.verbose)
        return args


class MNISTDefaultParser(CommonDefaultParser):

    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)
        if args.classifier:
            args.name = '_'.join([args.name, 'cls'])
        args.path = Path(args.path)/args.name
        return args


class XcoreTunedParser(CommonDefaultParser):
    def __init__(self, *args, defaults, **kwargs):
        super().__init__(*args, defaults=defaults, **kwargs)
        self.add_argument(
            '--xcore_tuned', action='store_true', default=False,
            help='Use a variation of the model tuned for xcore.ai.')

    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)
        if args.xcore_tuned:
            args.name = '_'.join([args.name, ('cls' if args.classifier else 'tuned')])
        args.path = Path(args.path)/args.name
        return args
