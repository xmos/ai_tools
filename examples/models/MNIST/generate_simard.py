#!/usr/bin/env python
#
# Copyright (c) 2020, XMOS Ltd, All rights reserved
from pathlib import Path
from mnist_common import MNISTModel, XcoreTunedParser
import tensorflow as tf

DEFAULT_PATH = Path(__file__).parent.joinpath('debug')
DEFAULT_NAME = 'simard'
DEFAULT_EPOCHS = 10
DEFAULT_BS = 64


class Simard(MNISTModel):
    def build(self):
        self._prep_backend()
        # Building
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.Input(shape=(29, 29, 1), name='input'),
                tf.keras.layers.Conv2D(5, kernel_size=5, strides=2,
                                       activation='relu', name='conv_1'),
                tf.keras.layers.Conv2D(50, kernel_size=5, strides=2,
                                       activation='relu', name='conv_2'),
                tf.keras.layers.Flatten(name='flatten'),
                tf.keras.layers.Dense(100, activation='relu', name='fc_1'),
                tf.keras.layers.Dense(10, activation='softmax', name='output')
            ])
        # Compilation
        self.core_model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
            metrics=['accuracy'])
        # Show summary
        self.core_model.summary()

    def prep_data(self):
        super().prep_data(simard_resize=True)

    def train(self, *, batch_size, save_history=True, **kwargs):
        # Image generator, # TODO: make this be optional with self._use_aug
        aug = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20, zoom_range=0.15,
            width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
            horizontal_flip=True, fill_mode="nearest")
        self.history = self.core_model.fit_generator(
            aug.flow(
                self.data['x_train'], self.data['y_train'], batch_size=batch_size),
            validation_data=(self.data['x_val'], self.data['y_val']),
            steps_per_epoch=len(self.data['x_train']) // batch_size,
            **kwargs)
        if save_history:
            self.save_training_history()


class SimardTuned(Simard):
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


def main(raw_args=None):
    parser = XcoreTunedParser(defaults={
        'batch_size': DEFAULT_BS,
        'epochs': DEFAULT_EPOCHS,
        'name': DEFAULT_NAME,
        'path': DEFAULT_PATH,
    })
    args = parser.parse_args(raw_args)
    kwargs = {
        'name': args.name,
        'path': args.path,
        'opt_classifier': args.classifier,
        'use_aug': args.augment_dataset
    }
    model = SimardTuned(**kwargs) if args.xcore_tuned else Simard(**kwargs)
    model.run(train_new_model=args.train_model,
              batch_size=args.batch, epochs=args.epochs)


if __name__ == "__main__":
    main()
