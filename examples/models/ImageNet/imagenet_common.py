# Copyright (c) 2020, XMOS Ltd, All rights reserved

import logging
import argparse
from tflite2xcore.model_generation import utils
from tflite2xcore.model_generation.interface import KerasClassifier
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np


class ImageNetModel(KerasClassifier):
    def __init__(self, *args, use_aug=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_aug = use_aug

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


def get_default_parser(**kwargs):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'path', nargs='?', default=None,
        help='Path to a directory where models and data will be saved in subdirectories.')
    parser.add_argument(
        '--use_gpu', action='store_true', default=False,
        help='Use GPU for training. Might result in non-reproducible results.')
    parser.add_argument(
        '--train_model', action='store_true', default=False,
        help='Train new model instead of loading pretrained tf.keras model.')
    parser.add_argument(
        '--classifier', action='store_true', default=False,
        help='Apply classifier optimizations during xcore conversion.')
    parser.add_argument(
        '-bs', '--batch', type=int, default=kwargs['DEFAULT_BS'],
        help='Batch size.')
    parser.add_argument(
        '-ep', '--epochs', type=int, default=kwargs['DEFAULT_EPOCHS'],
        help='Number of epochs.')
    parser.add_argument(
        '-aug', '--augment_dataset', action='store_true', default=False,
        help='Create a dataset with elastic transformations.')
    parser.add_argument(
        '-v', '--verbose', action='store_true', default=False,
        help='Verbose mode.')
    return parser


def run_main(model, *, train_new_model, epochs=None, batch_size=None):
    #if train_new_model:
    # Build model and compile
    model.build()
    # Prepare training data
    model.prep_data()
    # Train model
    #model.train(batch_size=batch_size, epochs=epochs)
    model.save_core_model()
    #else:
    # Recover previous state from file system
    #model.load_core_model()
    # Generate test data
    model.gen_test_data()
    # Populate converters
    model.populate_converters()
