# Copyright (c) 2020, XMOS Ltd, All rights reserved

# TODO: fix this hack
from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(realpath(__file__))))

from model_common import TrainableParser

import logging
import argparse
from pathlib import Path
from tflite2xcore.model_generation import utils
from tflite2xcore.model_generation.interface import KerasClassifier


#  ----------------------------------------------------------------------------
#                                  MODELS
#  ----------------------------------------------------------------------------

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
        self.data['export'] = self.data['x_test'][:10]
        self.data['quant'] = self.data['x_train'][:10]

    def run(self, train_new_model=False, epochs=None, batch_size=None):
        self._prep_backend()
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
        self.convert_and_save()


#  ----------------------------------------------------------------------------
#                                   PARSERS
#  ----------------------------------------------------------------------------

class MNISTDefaultParser(TrainableParser):

    def __init__(self, *args, defaults, **kwargs):
        super().__init__(*args, defaults=defaults, **kwargs)
        self.add_argument(
            '--name', type=str, default=defaults['name'],
            help="Name of the model, used in the creation of the model itself "
                 "and the target subdirectories."
        )
        self.add_argument(
            '--classifier', action='store_true', default=False,
            help='Apply classifier optimizations during xcore conversion.'
        )
        self.add_argument(
            '-aug', '--augment_dataset', action='store_true', default=False,
            help='Create a dataset with elastic transformations.'  # TODO: does this always mean elastic trf?
        )

    def _name_handler(self, args):
        if args.classifier:
            args.name = '_'.join([args.name, 'cls'])

    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)
        self._name_handler(args)
        args.path = Path(args.path)/args.name
        return args


class XcoreTunedParser(MNISTDefaultParser):
    def __init__(self, *args, defaults, **kwargs):
        super().__init__(*args, defaults=defaults, **kwargs)
        self.add_argument(
            '--xcore_tuned', action='store_true', default=False,
            help='Use a variation of the model tuned for xcore.ai.'
        )

    def _name_handler(self, args):
        if args.xcore_tuned:
            args.name = '_'.join([args.name, 'tuned'])
        super()._name_handler(args)
