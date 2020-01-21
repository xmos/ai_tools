# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import argparse
import logging
from pathlib import Path
import tensorflow as tf
import numpy as np

import os
import sys
# TODO: make sure we don't need this hack
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'interface_development')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model_development')))

import model_interface as mi
import tflite_utils
import model_tools as mt

DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'lenet5').resolve()
DEFAULT_EPOCHS = 10
DEFAULT_BS = 64


# Prepare data function
def prepare_lenet():
    x_train, x_test, x_val, y_train, y_test, y_val = mt.get_mnist(
        padding=2, categorical=True, flatten=False, y_float=False)
    x_train, y_train = mt.expand_dataset(
        x_train, y_train, 2, sigma=4.0, alpha=16.0)
    x_train, y_train = mt.shuffle(x_train, y_train)
    return {'x_train': np.float32(x_train[:3008]), 'x_test': np.float32(x_test[:500]), 'x_val': np.float32(x_val[:100]),
            'y_train': np.float32(y_train[:3008]), 'y_test': np.float32(y_test[:500]), 'y_val': np.float32(y_val[:100])}

# Broken bc of hyperparameters?
'''
generate_lenet5_tuned works and the only difference is:
- # filters in conv2d
- activation functions: relu instead of tanh
'''
class LeNet5(mi.KerasModel):

    def build(self):
        # Env, TODO: consider refactoring this to KerasModel
        tf.keras.backend.clear_session()
        tflite_utils.set_all_seeds()
        # Building
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.Input(shape=(32, 32, 1), name='input'),
                tf.keras.layers.Conv2D(filters=6,
                                       kernel_size=(5, 5),
                                       strides=1,
                                       activation='tanh',
                                       name='conv_1'),
                tf.keras.layers.AvgPool2D(pool_size=(2, 2),
                                          strides=2,
                                          name='avg_pool_1'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=16,
                                       kernel_size=(5, 5),
                                       strides=1,
                                       activation='tanh',
                                       name='conv_2'),
                tf.keras.layers.AvgPool2D(pool_size=(2, 2),
                                          strides=2,
                                          name='avg_pool_2'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=120,
                                       kernel_size=(5, 5),
                                       strides=1,
                                       activation='tanh',
                                       name='conv_3'),
                tf.keras.layers.Dense(units=84,
                                      activation='tanh',
                                      name='fc_1'),
                tf.keras.layers.Dense(units=10,
                                      activation='softmax',
                                      name='output')
            ]
        )
        opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-2 / 10)
        # 10 epochs with categorical data
        # Compilation
        self.core_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                      optimizer=opt, metrics=['accuracy'])
        # Show summary
        self.core_model.summary()

    # For training
    def prep_data(self):
        self.data = prepare_lenet()

    # For exports
    def gen_test_data(self):
        if not self.data:
            self.prep_data()
        self.data['export_data'] = self.data['x_test'][:10]
        self.data['quant'] = self.data['x_train'][:10]

    def train(self, BS, EPOCHS):
        # Image generator
        aug = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20, zoom_range=0.15,
            width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
            horizontal_flip=True, fill_mode="nearest")
        # Train the network
        history_lenet = self.core_model.fit_generator(
            aug.flow(self.data['x_train'], self.data['y_train'], batch_size=BS),
            validation_data=(self.data['x_test'], self.data['y_test']),
            steps_per_epoch=len(self.data['x_train']) // BS,
            epochs=EPOCHS)


def main(path=DEFAULT_PATH, train_new_model=False,
         BS=DEFAULT_BS, EPOCHS=DEFAULT_EPOCHS):
    lenet = LeNet5('lenet5', path)
    if train_new_model:
        # Build model and compile
        lenet.build()
        # Prepare training data
        lenet.prep_data()
        # Train model
        lenet.train(BS, EPOCHS)
        lenet.save_core_model()
    else:
        # Recover previous state from file system
        lenet.load_core_model()
    # Generate test data
    lenet.gen_test_data()
    # Populate converters
    lenet.populate_converters()


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
        '--batch', type=int, default=DEFAULT_BS,
        help='Batch size.')
    parser.add_argument(
        '--epochs', type=int, default=DEFAULT_EPOCHS,
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
         BS=args.batch,
         EPOCHS=args.epochs)
