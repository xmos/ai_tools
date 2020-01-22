# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import sys
import argparse
import logging
from pathlib import Path
import tensorflow as tf
import numpy as np

sys.path.append('../interface_development/')
sys.path.append('../model_development/')
import model_interface as mi
import tflite_utils
import model_tools as mt

from generate_lenet5 import LeNet5


DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'lenet5_tuned').resolve()
DEFAULT_EPOCHS = 10
DEFAULT_BS = 64
DEFAULT_AUG = False


class LeNet5Tuned(LeNet5):

    def build(self):
        super().build()
        # Building
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.Input(shape=(32, 32, 1), name='input'),

                tf.keras.layers.Conv2D(8, kernel_size=5, name='conv_1'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),

                tf.keras.layers.AvgPool2D(pool_size=2, strides=2, name='avg_pool_1'),

                tf.keras.layers.Conv2D(16, kernel_size=5, name='conv_2'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),

                tf.keras.layers.AvgPool2D(pool_size=2, strides=2, name='avg_pool_2'),

                tf.keras.layers.Conv2D(128, kernel_size=5, name='conv_3'),
                tf.keras.layers.ReLU(),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(96, activation='relu', name='fc_1'),
                tf.keras.layers.Dense(10, activation='softmax', name='output')
            ]
        )
        opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-2 / 10)
        # 10 epochs with categorical data
        # Compilation
        self.core_model.compile(loss='sparse_categorical_crossentropy',
                                optimizer=opt, metrics=['accuracy'])
        # Show summary
        self.core_model.summary()


def main(path=DEFAULT_PATH, train_new_model=False,
         batch_size=DEFAULT_BS, epochs=DEFAULT_EPOCHS,
         use_aug=DEFAULT_AUG):
    lenet5_tuned = LeNet5Tuned('lenet5_tuned', path)
    if train_new_model:
        # Build model and compile
        lenet5_tuned.build()
        # Prepare training data
        lenet5_tuned.prep_data(use_aug)
        # Train model
        lenet5_tuned.train(batch_size=batch_size, epochs=epochs)
        lenet5_tuned.save_core_model()
    else:
        # Recover previous state from file system
        lenet5_tuned.load_core_model()
    # Generate test data
    lenet5_tuned.gen_test_data(use_aug)
    # Populate converters
    lenet5_tuned.populate_converters()


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
        '-aug', '--augment_dataset', action='store_true', default=False,
        help='Create a dataset with elastic transformations.')
    parser.add_argument(
        '-bs', '--batch', type=int, default=DEFAULT_BS,
        help='Batch size.')
    parser.add_argument(
        '-ep', '--epochs', type=int, default=DEFAULT_EPOCHS,
        help='Number of epochs.')
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
         batch_size=args.batch, epochs=args.epochs,
         use_aug=args.augment_dataset)
