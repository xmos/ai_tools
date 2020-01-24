# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import argparse
import logging
from pathlib import Path
import tensorflow as tf
import os
import sys
# TODO: make sure we don't need this hack
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'interface_development')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model_development')))

from generate_simrad import Simrad

import model_interface as mi
import tflite_utils
import model_tools as mt

DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'simrad_tuned').resolve()
DEFAULT_EPOCHS = 10
DEFAULT_BS = 64
DEFAULT_AUG = False


class SimradTuned(Simrad):
    def build(self):
        self._prep_backend()
        # Building
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.Input(shape=(29, 29, 1), name='input'),
                tf.keras.layers.Conv2D(8, kernel_size=5, strides=2,
                                       activation='relu', name='conv_1'),
                tf.keras.layers.Conv2D(64, kernel_size=5, strides=2,
                                       activation='relu', name='conv_2'),
                tf.keras.layers.Flatten(name='flatten'),
                tf.keras.layers.Dense(96, activation='relu', name='fc_1'),
                tf.keras.layers.Dense(10, activation='softmax', name='output')
            ])
        # Compilation
        self.core_model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
            metrics=['accuracy'])
        # Show summary
        self.core_model.summary()


def main(path=DEFAULT_PATH, train_new_model=False,
         batch_size=DEFAULT_BS, epochs=DEFAULT_EPOCHS,
         use_aug=DEFAULT_AUG):

    simrad = SimradTuned('simrad_tuned', path)

    if train_new_model:
        # Build model and compile
        simrad.build()
        # Prepare training data
        simrad.prep_data(use_aug)
        # Train model
        simrad.train(batch_size=batch_size, epochs=epochs, save_history=True)
        simrad.save_core_model()
    else:
        # Recover previous state from file system
        simrad.load_core_model()
    # Generate test data
    simrad.gen_test_data(use_aug)
    # Populate converters
    simrad.populate_converters()


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
        '--train_model', action='store_true', default=False,
        help='Train new model instead of loading pretrained tf.keras model.')
    parser.add_argument(
        '-bs', '--batch', type=int, default=DEFAULT_BS,
        help='Batch size.')
    parser.add_argument(
        '-ep', '--epochs', type=int, default=DEFAULT_EPOCHS,
        help='Number of epochs.')
    parser.add_argument(
        '-aug', '--augment_dataset', action='store_true', default=False,
        help='Create a dataset with elastic transformations.')
    parser.add_argument(
        '-v', '--verbose', action='store_true', default=False,
        help='Verbose mode.')
    args = parser.parse_args()

    verbose = args.verbose
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.info(f"Eager execution enabled: {tf.executing_eagerly()}")
    tflite_utils.set_gpu_usage(args.use_gpu, verbose)

    main(path=args.path,
         train_new_model=args.train_model,
         batch_size=args.batch,
         epochs=args.epochs,
         use_aug=args.augment_dataset)
