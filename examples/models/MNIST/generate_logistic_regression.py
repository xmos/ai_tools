#!/usr/bin/env python
#
# Copyright (c) 2020, XMOS Ltd, All rights reserved
from pathlib import Path
from mnist_common import MNISTModel, MNISTDefaultParser
import tensorflow as tf
from tensorflow.keras import layers

DEFAULT_PATH = Path(__file__).parent.joinpath('debug')
DEFAULT_NAME = 'logistic_regression'
DEFAULT_EPOCHS = 30
DEFAULT_BS = 64


class LogisticRegression(MNISTModel):

    def build(self):
        self._prep_backend()
        # Building
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                layers.Flatten(input_shape=(28, 28, 1), name='input'),
                layers.Dense(10, activation='softmax',
                             kernel_regularizer=tf.keras.regularizers.l1(2e-5))
            ]
        )
        # Compilation
        self.core_model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        # Show summary
        self.core_model.summary()

    def prep_data(self):
        super().prep_data(padding=0)


def main(raw_args=None):
    parser = MNISTDefaultParser(defaults={
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
    model = LogisticRegression(**kwargs)
    model.run(train_new_model=args.train_model,
              batch_size=args.batch_size, epochs=args.epochs)


if __name__ == '__main__':
    main()
