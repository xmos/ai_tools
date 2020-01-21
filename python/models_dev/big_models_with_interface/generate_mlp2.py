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

import model_interface as mi
import tflite_utils
import model_tools as mt

DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'mlp2').resolve()
DEFAULT_EPOCHS = 10
DEFAULT_BS = 64
DEFAULT_AUG = False


class MLP2(mi.KerasModel):
    def build(self):
        super().build()
        # Building
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.layers.Flatten(input_shape=(32, 32, 1), name='input'),
                tf.keras.layers.Dense(416, activation='tanh', name='dense_1'),
                tf.keras.layers.Dense(288, activation='tanh', name='dense_2'),
                tf.keras.layers.Dense(10, activation='softmax', name='output')
            ])
        # Compilation
        self.core_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
            metrics=['accuracy'])
        # Show summary
        self.core_model.summary()

    # For training
    def prep_data(self, aug=False):
        self.data = mt.prepare_MLP(aug)

    # For exports
    def gen_test_data(self, aug=False):
        if not self.data:
            self.prep_data(aug)
        self.data['export_data'] = self.data['x_test'][:10]
        self.data['quant'] = self.data['x_train'][:10]

    def train(self, BS, EPOCHS):
        # Multi Layer Perceptron 1
        history_mlp2 = self.core_model.fit(
            self.data['x_train'], self.data['y_train'],
            batch_size=BS, epochs=EPOCHS,
            validation_data=(self.data['x_test'], self.data['y_test']))


def main(path=DEFAULT_PATH, train_new_model=False,
         BS=DEFAULT_BS, EPOCHS=DEFAULT_EPOCHS,
         AUG=DEFAULT_AUG):

    mlp2 = MLP2('mlp2', path)

    if train_new_model:
        # Build model and compile
        mlp2.build()
        # Prepare training data
        mlp2.prep_data(AUG)
        # Train model
        mlp2.train(BS, EPOCHS)
        mlp2.save_core_model()
    else:
        # Recover previous state from file system
        mlp2.load_core_model()
    # Generate test data
    mlp2.gen_test_data(AUG)
    # Populate converters
    mlp2.populate_converters()


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
         BS=args.batch,
         EPOCHS=args.epochs,
         AUG=args.augment_dataset)
