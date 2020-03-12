# Copyright (c) 2020, XMOS Ltd, All rights reserved

import logging
import argparse
from pathlib import Path
from tflite2xcore.model_generation import utils
from tflite2xcore.model_generation.interface import KerasClassifier
import tensorflow as tf
import numpy as np


class ImageNetModel(KerasClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prep_data(self, *, simard_resize=False, padding=2):
        image_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg"
        f = tf.keras.utils.get_file("grace_hopper.jpg", image_url)
        img = tf.keras.preprocessing.image.load_img(f, target_size=[159, 159])
        x = tf.keras.preprocessing.image.img_to_array(img)
        example_tensor = np.expand_dims(x, axis=0)
        self.data['x_test'] = example_tensor
        self.data['x_train'] = example_tensor
        for k, v in self.data.items():
            logging.debug(f"Prepped data[{k}] with shape: {v.shape}")

    def gen_test_data(self):
        if not self.data:
            self.prep_data()
        self.data['export_data'] = self.data['x_test']
        self.data['quant'] = self.data['x_train']

    def run(self):
        # Build model and compile
        self.build()
        # Prepare training data
        self.prep_data()
        self.save_core_model()
        # Generate test data
        self.gen_test_data()
        # Populate converters
        self.populate_converters()


class ImagenetDefaultParser(argparse.ArgumentParser):

    def __init__(self, *args, defaults, **kwargs):
        kwargs["formatter_class"] = argparse.ArgumentDefaultsHelpFormatter
        super().__init__(*args, **kwargs)
        self.add_argument(
            'path', nargs='?', default=defaults['path'],
            help='Path to a directory where models and data will be saved in subdirectories.')
        self.add_argument(
            '-v', '--verbose', action='store_true', default=False,
            help='Verbose mode.')

    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)
        utils.set_verbosity(args.verbose)
        return args
