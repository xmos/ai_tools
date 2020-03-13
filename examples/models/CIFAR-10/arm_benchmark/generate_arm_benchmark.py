#!/usr/bin/env python
#
# Copyright (c) 2020, XMOS Ltd, All rights reserved
import argparse
import logging
from pathlib import Path
import numpy as np
from tflite2xcore.model_generation import utils
from tflite2xcore.model_generation.interface import KerasClassifier
import tensorflow as tf

DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'arm_benchmark').resolve()
DEFAULT_EPOCHS = 30
DEFAULT_BS = 32


# TODO refactor and rename appropriately
def get_normalized_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    scale = tf.constant(255, dtype=tf.dtypes.float32)
    x_train, x_test = train_images/scale - .5, test_images/scale - .5
    y_train, y_test = train_labels, test_labels

    return {'x_train': np.float32(x_train), 'y_train': np.float32(y_train),
            'x_test': np.float32(x_test), 'y_test': np.float32(y_test)}


class ArmBenchmark(KerasClassifier):

    def build(self):
        self._prep_backend()
        # Building
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.layers.Conv2D(filters=32, kernel_size=5,
                                       padding='same', input_shape=(32, 32, 3)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

                tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

                tf.keras.layers.Conv2D(filters=64, kernel_size=5, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(10, activation='softmax')
            ]
        )

        # Compilation
        self.core_model.compile(loss='sparse_categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
        # Show summary
        self.core_model.summary()

    # For training
    def prep_data(self):
        self.data = get_normalized_data()
        for k, v in self.data.items():
            logging.debug(f"Prepped data[{k}] with shape: {v.shape}")

    # For exports
    def gen_test_data(self):
        if not self.data:
            self.prep_data()

        sorted_inds = np.argsort(self.data['y_test'], axis=0, kind='mergesort')
        subset_inds = np.searchsorted(self.data['y_test'][sorted_inds].flatten(), np.arange(10))  # pylint: disable=unsubscriptable-object
        subset_inds = sorted_inds[subset_inds]
        self.data['export'] = self.data['x_test'][subset_inds.flatten()]  # pylint: disable=unsubscriptable-object
        self.data['quant'] = self.data['x_train']

    def train(self, *, batch_size, save_history=True, **kwargs):
        # Image generator, # TODO: make this be optional with use_aug arg
        aug = tf.keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)
        aug.fit(self.data['x_train'])

        # Train the network
        self.history = self.core_model.fit_generator(
            aug.flow(
                self.data['x_train'], self.data['y_train'], batch_size=batch_size),
            validation_data=(self.data['x_test'], self.data['y_test']),
            steps_per_epoch=len(self.data['x_train']) // batch_size,
            **kwargs)
        if save_history:
            self.save_training_history()


def main(path=DEFAULT_PATH, train_new_model=False,
         batch_size=DEFAULT_BS, epochs=DEFAULT_EPOCHS,
         opt_classifier=False):

    arm_benchmark = ArmBenchmark('arm_benchmark', path, opt_classifier=opt_classifier)

    if train_new_model:
        # Build model and compile
        arm_benchmark.build()
        # Prepare training data
        arm_benchmark.prep_data()
        # Train model
        arm_benchmark.train(batch_size=batch_size, epochs=epochs)
        arm_benchmark.save_core_model()
    else:
        # Recover previous state from file system
        arm_benchmark.load_core_model()
    arm_benchmark.convert_and_save()


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
        '--classifier', action='store_true', default=False,
        help='Apply classifier optimizations during xcore conversion.')
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

    utils.set_verbosity(args.verbose)
    utils.set_gpu_usage(args.use_gpu, args.verbose)

    main(path=args.path,
         train_new_model=args.train_model,
         batch_size=args.batch,
         epochs=args.epochs,
         opt_classifier=args.classifier)
