#!/usr/bin/env python
#
# Copyright (c) 2020, XMOS Ltd, All rights reserved
from pathlib import Path
from mnist_common import MNISTModel, XcoreTunedParser
import tensorflow as tf

DEFAULT_PATH = Path(__file__).parent.joinpath('debug')
DEFAULT_NAME = 'mlp'
DEFAULT_EPOCHS = 10
DEFAULT_BS = 64


class MLP(MNISTModel):
    def build(self):
        self._prep_backend()
        # Building
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.layers.Flatten(input_shape=(32, 32, 1), name='input'),
                tf.keras.layers.Dense(390, activation='tanh', name='dense_1'),
                tf.keras.layers.Dense(290, activation='tanh', name='dense_2'),
                tf.keras.layers.Dense(10, activation='softmax', name='output')
            ])
        # Compilation
        self.core_model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
            metrics=['accuracy'])
        # Show summary
        self.core_model.summary()


class MLPTuned(MLP):
    def build(self):
        self._prep_backend()
        # Building
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.layers.Flatten(input_shape=(32, 32, 1), name='input'),
                tf.keras.layers.Dense(384, activation='relu', name='dense_1'),
                tf.keras.layers.Dense(256, activation='relu', name='dense_2'),
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
    model = MLPTuned(**kwargs) if args.xcore_tuned else MLP(**kwargs)
    model.run(train_new_model=args.train_model,
              batch_size=args.batch_size, epochs=args.epochs)


if __name__ == "__main__":
    main()
