# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import argparse
import logging
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import model_interface as mi
import tflite_utils

DEFAULT_CLASSES = 4
DEFAULT_INPUTS = 32


# Prepare data function
def generate_fake_lin_sep_dataset(classes=2, dim=32, *,
                                  train_samples_per_class=5120,
                                  test_samples_per_class=1024):
    z = np.linspace(0, 2*np.pi, dim)

    # generate data and class labels
    x_train, x_test, y_train, y_test = [], [], [], []
    for j in range(classes):
        mean = np.sin(z) + 10*j/classes
        cov = 10 * np.diag(.5*np.cos(j * z) + 2) / (classes-1)
        x_train.append(
            np.random.multivariate_normal(
                mean, cov, size=train_samples_per_class))
        x_test.append(
            np.random.multivariate_normal(
                mean, cov, size=test_samples_per_class))
        y_train.append(j * np.ones((train_samples_per_class, 1)))
        y_test.append(j * np.ones((test_samples_per_class, 1)))

    # stack arrays
    x_train = np.vstack(x_train)
    y_train = np.vstack(y_train)
    x_test = np.vstack(x_test)
    y_test = np.vstack(y_test)

    # normalize
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # expand dimensions for TFLite compatibility
    def expand_array(arr):
        return np.reshape(arr, arr.shape + (1, 1))
    x_train = expand_array(x_train)
    x_test = expand_array(x_test)

    return {'x_train': np.float32(x_train), 'y_train': np.float32(y_train),
            'x_test': np.float32(x_test), 'y_test': np.float32(y_test)}


# Class for the model
class FcDeepinShallowoutFinal(mi.KerasModel):
    def build(self):  # add keyboard optimizer, loss and metrics???
        input_dim = self.input_dim
        output_dim = self.output_dim
        # Env
        tf.keras.backend.clear_session()
        tflite_utils.set_all_seeds()
        # Building
        model = tf.keras.Sequential(name=self.name)
        model.add(layers.Flatten(input_shape=(input_dim, 1, 1), name='input'))
        model.add(layers.Dense(output_dim,
                               activation='softmax', name='ouptut'))
        # Compilation
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        # Add to dict
        self.models[self.name] = model
        # Show summary
        model.summary()

    # For training
    def prep_data(self):
        self.data = generate_fake_lin_sep_dataset(
            self.output_dim, self.input_dim,
            train_samples_per_class=51200//self.output_dim,
            test_samples_per_class=10240//self.output_dim)

    # For exports
    def gen_test_data(self):
        if not self.data:
            self.prep_data()
        subset_inds = np.searchsorted(
            self.data['y_test'].flatten(), np.arange(self.output_dim))
        self.data['export_data'] = self.data['x_test'][subset_inds]
        self.data['quant'] = self.data['x_train']

    def train(self):
        super().train(128, 5*(self.output_dim-1))  # BS and EPOCHS


def main(input_dim=DEFAULT_INPUTS, classes=DEFAULT_CLASSES, *,
         train_new_model=False):
    # Instantiate model
    test_model = FcDeepinShallowoutFinal(
        'fc_deepin_shallowout_final', Path('./fc_deepin_shallowout_final'),
        input_dim, classes)
    if train_new_model:
        # Build model and compile
        test_model.build()
        # Prepare training data
        test_model.prep_data()
        # Train model
        test_model.train()
        test_model.save_core_model()
    else:
        # Recover previous state from file system
        test_model.load_core_model()
    # Generate test data
    test_model.gen_test_data()
    # Populate converters
    test_model.populate_converters()
    # test_model.to_tf_float()
    # test_model.to_tf_quant()
    # test_model.to_tf_stripped()
    # Convert and save, will break in stripped
    test_model.convert_and_save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--use_gpu', action='store_true', default=False,
        help='Use GPU for training. Might result in non-reproducible results')
    parser.add_argument(
        '--classes', type=int, default=DEFAULT_CLASSES,
        help='Number of classes, must be between 2 and 15.')
    parser.add_argument(
        '--inputs', type=int, default=DEFAULT_INPUTS,
        help='Input dimension, must be multiple of 32.')
    parser.add_argument(
        '--train_model', action='store_true', default=False,
        help='Train new model instead of loading pretrained tf.keras model.')
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
    main(input_dim=args.inputs, classes=args.classes,
         train_new_model=args.train_model)