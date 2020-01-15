import os
import shutil
import pathlib
import argparse
import logging
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from termcolor import colored

import model_interface as mi
import tflite_utils


class FcDeepinShallowoutFinal(mi.KerasModel):

    def generate_fake_lin_sep_dataset(self, classes=2, dim=32, *,
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

    # add keyboard optimizer, loss and metrics???
    def build(self, input_dim, out_dim=2):
        input_dim = self.input_dim
        output_dim = self.output_dim
        # Env
        tf.keras.backend.clear_session()
        tflite_utils.set_all_seeds()
        # Building
        self.core_model = tf.keras.Sequential(name=self.name)
        self.core_model.add(layers.Flatten(input_shape=(input_dim, 1, 1),
                                           name='input'))
        self.core_model.add(layers.Dense(output_dim, activation='softmax',
                                         name='ouptut'))
        # Compilation
        self.core_model.compile(optimizer='adam',
                                loss='sparse_categorical_crossentropy',
                                metrics=['accuracy'])
        # Show summary
        self.core_model.summary()

    def prep_data(self):
        self.data = self.generate_fake_lin_sep_dataset(
            self.output_dim,
            self.input_dim,
            train_samples_per_class=51200//self.output_dim,
            test_samples_per_class=10240//self.output_dim)

    def gen_test_data(self):
        if not self.data:
            self.prep_data()
        subset_inds = np.searchsorted(self.data['y_test'].flatten(),
                                      np.arange(self.output_dim))
        self.data['export_data'] = self.data['x_test'][subset_inds]
        self.data['quant'] = self.data['x_train']

    def train(self):
        self.BS = 128
        self.EPOCHS = 5*(self.output_dim-1)
        super().train(self.BS, self.EPOCHS)


def printc(*s, c='green', back='on_grey'):
    if len(s) == 1:
        print(colored(str(s)[2:-3], c, back))
    else:
        print(colored(s[0], c, back), str(s[1:])[1:-2])


def debug_dir(path, name, before):
    if before:
        printc(name + ' directory before generation:')
    else:
        printc(name + ' directory after generation:')
    print([str(x.name) for x in path.iterdir() if x.is_file() or x.is_dir()])


def debug_keys_header(title, test_model):
    printc(title, c='blue')
    debug_keys('Model keys:\n', test_model.models)
    debug_keys('Data keys:\n', test_model.data)
    debug_keys('Converter keys:\n', test_model.converters)


def debug_keys(string, dic):
    printc(string, dic.keys())


def debug_conv(to_type, test_model, datapath, modelpath):
    debug_keys_header('Conversion to ' + to_type + ' start', test_model)
    debug_dir(modelpath, 'Models', True)
    printc('Converting model...', c='yellow')
    choose_conv_or_save(to_type, test_model, False)
    debug_dir(modelpath, 'Models', False)
    debug_dir(datapath, 'Data', True)
    printc('Saving data...', c='yellow')
    choose_conv_or_save(to_type, test_model, True)
    debug_dir(datapath, 'Data', False)


def choose_conv_or_save(conv, test_model, save):
    if not save:
        return{
            'float': lambda m: m.to_tf_float(),
            'quant': lambda m: m.to_tf_quant(),
            'stripped': lambda m: m.to_tf_stripped(),
            'xcore': lambda m: m.to_tf_xcore()
        }[conv](test_model)
    else:
        return{
            'float': lambda m: m.save_tf_float_data(),
            'quant': lambda m: m.save_tf_quant_data(),
            'stripped': lambda m: m.save_tf_stripped_data(),
            'xcore': lambda m: m.save_tf_xcore_data()
        }[conv](test_model)


def main():
    # Random seed
    random.seed(42)
    # Remove everthing
    if os.path.exists('./debug/keras_test'):
        shutil.rmtree('./debug/keras_test')
    modelpath = pathlib.Path('./debug/keras_test/models')
    datapath = pathlib.Path('./debug/keras_test/test_data')
    # Instantiation
    test_model = FcDeepinShallowoutFinal(
        'fc_deepin_shallowout_final', pathlib.Path('./debug/keras_test'), 32, 2)
    printc('Model name property:\n', test_model.name)
    # Build
    debug_keys_header('Keys before build()', test_model)
    debug_dir(modelpath, 'Models', True)
    debug_dir(datapath, 'Data', True)
    test_model.build(32)
    # Train data preparation
    test_model.prep_data()
    debug_keys_header('Keys after build() and prep_data()', test_model)
    debug_dir(modelpath, 'Models', False)
    debug_dir(datapath, 'Data', False)
    # Training
    printc('Training:', c='blue')
    test_model.train()
    # Save model
    printc('Saving model', c='blue')
    test_model.save_core_model()
    debug_dir(modelpath, 'Models', False)
    debug_dir(datapath, 'Data', False)
    # Export data generation
    printc('Generating export data', c='blue')
    test_model.gen_test_data()
    debug_keys('Data keys after export data generation:\n', test_model.data)
    # Conversions
    debug_conv('float', test_model, datapath, modelpath)
    debug_conv('quant', test_model, datapath, modelpath)
    debug_conv('stripped', test_model, datapath, modelpath)
    debug_conv('xcore', test_model, datapath, modelpath)
    # Final status
    debug_keys_header('Final status', test_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Verbose mode.')
    args = parser.parse_args()

    verbose = args.verbose
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        logging.info(f"Eager execution enabled: {tf.executing_eagerly()}")

    main()