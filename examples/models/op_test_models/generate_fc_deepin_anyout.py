#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import argparse
from pathlib import Path
import numpy as np
from tflite2xcore.model_generation import utils
from tflite2xcore.model_generation.interface import KerasModel
import tensorflow as tf
import op_test_models_common as common

DEFAULT_OUTPUT_DIM = 10
DEFAULT_INPUT_DIM = 32
DEFAULT_EPOCHS = 10
DEFAULT_BS = 64
DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'fc_deepin_anyout').resolve()


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


class FcDeepinAnyout(KerasModel):
    def build(self, input_dim, output_dim, *, bias_init, weight_init):
        super().build()

        # Building
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.layers.Flatten(input_shape=(input_dim, 1, 1)),
                tf.keras.layers.Dense(output_dim, activation='softmax',
                                      bias_initializer=bias_init,
                                      kernel_initializer=weight_init)
            ]
        )
        # Compilation
        self.core_model.compile(optimizer='adam',
                                loss='sparse_categorical_crossentropy',
                                metrics=['accuracy'])
        # Show summary
        self.core_model.summary()

    @property
    def input_dim(self):
        return self.input_shape[0]

    @property
    def output_dim(self):
        return self.output_shape[0]

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
        super().train(batch_size=128, epochs=5*(self.output_dim-1))

    def to_tf_stripped(self):
        super().to_tf_stripped(remove_softmax=True)

    def to_tf_xcore(self):
        super().to_tf_xcore(remove_softmax=True)


def main(path=DEFAULT_PATH, *,
         input_dim=DEFAULT_INPUT_DIM, output_dim=DEFAULT_OUTPUT_DIM,
         train_new_model=False,
         bias_init=common.DEFAULT_CONST_INIT, weight_init=common.DEFAULT_UNIF_INIT):

    # Instantiate model
    test_model = FcDeepinAnyout('fc_deepin_anyout', Path(path))
    if train_new_model:
        # Build model and compile
        test_model.build(input_dim, output_dim,
                         bias_init=bias_init, weight_init=weight_init)
        # Prepare training data
        test_model.prep_data()
        # Train model
        test_model.train()
        test_model.save_core_model()
    else:
        # Recover previous state from file system
        test_model.load_core_model()
        if output_dim != test_model.output_dim:
            raise ValueError(
                f"specified output_dim ({output_dim}) "
                f"does not match loaded model's output_dim ({test_model.output_dim})"
            )
    # Generate test data
    test_model.gen_test_data()
    # Populate converters
    test_model.populate_converters()


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
        '-out', '--output_dim', type=int, default=DEFAULT_OUTPUT_DIM,
        help='Number of output dimensions, must be at least 2.')
    parser.add_argument(
        '-in', '--input_dim', type=int, default=DEFAULT_INPUT_DIM,
        help='Input dimension, must be multiple of 32.')
    parser.add_argument(
        '--train_model', action='store_true', default=False,
        help='Train new model instead of loading pretrained tf.keras model.')
    parser.add_argument(
        '-v', '--verbose', action='store_true', default=False,
        help='Verbose mode.')
    parser = common.parser_add_initializers(parser)
    args = parser.parse_args()

    utils.set_verbosity(args.verbose)
    utils.set_gpu_usage(args.use_gpu, args.verbose)
    
    initializers = common.initializer_args_handler(args)

    main(path=args.path,
         input_dim=args.input_dim, output_dim=args.output_dim,
         train_new_model=args.train_model,
         bias_init=initializers['bias_init'],
         weight_init=initializers['weight_init'])
