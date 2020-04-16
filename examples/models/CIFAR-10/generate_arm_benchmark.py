#!/usr/bin/env python
#
# Copyright (c) 2020, XMOS Ltd, All rights reserved

# TODO: fix this hack
from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(realpath(__file__))))

from model_common import TrainableParser

from pathlib import Path
import numpy as np
from tflite2xcore.utils import VerbosityParser
from tflite2xcore.model_generation.cifar10 import get_normalized_data
from tflite2xcore.model_generation.interface import KerasModel
import tensorflow as tf

DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'arm_benchmark').resolve()
DEFAULT_EPOCHS = 30
DEFAULT_BS = 32


class ArmBenchmark(KerasModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aug = None

    def build(self):
        self._prep_backend()
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

        self.core_model.compile(loss='sparse_categorical_crossentropy',
                                optimizer='adam', metrics=['accuracy'])
        self.core_model.summary()

    def prep_data(self, use_aug=True):
        self.data = get_normalized_data()
        for k, v in self.data.items():
            self.logger.debug(f"Prepped data[{k}] with shape: {v.shape}")

        if use_aug:
            self.logger.debug(f"Prepped data[{k}] with shape: {v.shape}")
            self.aug = tf.keras.preprocessing.image.ImageDataGenerator(
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
            self.aug.fit(self.data['x_train'])

    def gen_test_data(self):
        if not self.data:
            self.prep_data()

        sorted_inds = np.argsort(self.data['y_test'], axis=0, kind='mergesort')
        subset_inds = np.searchsorted(
            self.data['y_test'][sorted_inds].flatten(),  # pylint: disable=unsubscriptable-object
            np.arange(10)
        )
        subset_inds = sorted_inds[subset_inds]
        self.data['export'] = self.data['x_test'][subset_inds.flatten()]  # pylint: disable=unsubscriptable-object
        self.data['quant'] = self.data['x_train']

    def train(self, *, batch_size, epochs, save_history=True):
        fit_args = dict(
            validation_data=(self.data['x_test'], self.data['y_test']),
            epochs=epochs
        )
        if self.aug:
            fit_args['x'] = self.aug.flow(
                self.data['x_train'], self.data['y_train'],
                batch_size=batch_size
            )
        else:
            fit_args['x'] = self.data['x_train']
            fit_args['y'] = self.data['y_train']
            fit_args['batch_size'] = batch_size

        self.history = self.core_model.fit(**fit_args)
        if save_history:
            self.save_training_history()


def main(raw_args=None):
    parser = TrainableParser(defaults={
        'batch_size': DEFAULT_BS,
        'epochs': DEFAULT_EPOCHS,
        'path': DEFAULT_PATH,
    })
    args = parser.parse_args(raw_args)

    model = ArmBenchmark('arm_benchmark', args.path)

    if args.train_model:
        model.build()
        model.prep_data()
        model.train(batch_size=args.batch_size, epochs=args.epochs)
        model.save_core_model()
    else:
        model.load_core_model()
    model.convert_and_save()


if __name__ == "__main__":
    main()
