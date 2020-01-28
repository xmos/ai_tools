#!/usr/bin/env python
#
# Copyright (c) 2020, XMOS Ltd, All rights reserved
from pathlib import Path
from tflite2xcore.model_generation import utils
from mnist_common import MNISTModel, get_default_parser, run_main
import tensorflow as tf

DEFAULT_PATH = {
    'lenet5': Path(__file__).parent.joinpath('debug', 'lenet5').resolve(),
    'lenet5_tuned': Path(__file__).parent.joinpath('debug', 'lenet5_tuned').resolve(),
    'lenet5_cls': Path(__file__).parent.joinpath('debug', 'lenet5_cls').resolve()
}
DEFAULT_EPOCHS = 10
DEFAULT_BS = 64


# Broken in TensorFlow 2.0, solved in Tensorflow 2.1
# Issue: 'tanh' before AvgPool2D
class LeNet5(MNISTModel):

    def build(self):
        self._prep_backend()
        # Building
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.Input(shape=(32, 32, 1), name='input'),

                tf.keras.layers.Conv2D(6, kernel_size=5, name='conv_1'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('tanh'),

                tf.keras.layers.AvgPool2D(pool_size=2, strides=2, name='avg_pool_1'),

                tf.keras.layers.Conv2D(16, kernel_size=5, name='conv_2'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('tanh'),

                tf.keras.layers.AvgPool2D(pool_size=2, strides=2, name='avg_pool_2'),

                tf.keras.layers.Conv2D(120, kernel_size=5, name='conv_3'),
                tf.keras.layers.Activation('tanh'),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(84, activation='tanh', name='fc_1'),
                tf.keras.layers.Dense(10, activation='softmax', name='output')
            ]
        )
        opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-2 / 10)
        # 10 epochs with categorical data
        # Compilation
        self.core_model.compile(loss='sparse_categorical_crossentropy',
                                optimizer=opt, metrics=['accuracy'])
        # Show summary
        self.core_model.summary()

    def train(self, *, batch_size, save_history=True, **kwargs):
        # Image generator, # TODO: make this be optional with self._use_aug
        aug = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20, zoom_range=0.15,
            width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
            horizontal_flip=True, fill_mode="nearest")
        # Train the network
        self.history = self.core_model.fit_generator(
            aug.flow(
                self.data['x_train'], self.data['y_train'], batch_size=batch_size),
            validation_data=(self.data['x_test'], self.data['y_test']),
            steps_per_epoch=len(self.data['x_train']) // batch_size,
            **kwargs)
        if save_history:
            self.save_training_history()


class LeNet5Tuned(LeNet5):

    def build(self):
        self._prep_backend()
        # Building
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.Input(shape=(32, 32, 1), name='input'),

                tf.keras.layers.Conv2D(8, kernel_size=5, name='conv_1'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),

                tf.keras.layers.AvgPool2D(pool_size=2, strides=2, name='avg_pool_1'),

                tf.keras.layers.Conv2D(16, kernel_size=5, name='conv_2'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),

                tf.keras.layers.AvgPool2D(pool_size=2, strides=2, name='avg_pool_2'),

                tf.keras.layers.Conv2D(128, kernel_size=5, name='conv_3'),
                tf.keras.layers.ReLU(),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(96, activation='relu', name='fc_1'),
                tf.keras.layers.Dense(10, activation='softmax', name='output')
            ]
        )
        opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-2 / 10)
        # 10 epochs with categorical data
        # Compilation
        self.core_model.compile(loss='sparse_categorical_crossentropy',
                                optimizer=opt, metrics=['accuracy'])
        # Show summary
        self.core_model.summary()


def main(path=None, train_new_model=False,
         batch_size=DEFAULT_BS, epochs=DEFAULT_EPOCHS,
         use_aug=False, xcore_tuned=False, opt_classifier=False):

    name = ('lenet5_cls' if opt_classifier else 'lenet5_tuned') if xcore_tuned else 'lenet5'
    kwargs = {
        'name': name,
        'path': path if path else DEFAULT_PATH[name],
        'opt_classifier': opt_classifier,
        'use_aug': use_aug
    }

    run_main(
        model=LeNet5Tuned(**kwargs) if xcore_tuned else LeNet5(**kwargs),
        train_new_model=train_new_model,
        batch_size=batch_size, epochs=epochs
    )


if __name__ == "__main__":
    parser = get_default_parser(DEFAULT_BS=DEFAULT_BS, DEFAULT_EPOCHS=DEFAULT_EPOCHS)
    parser.add_argument(
        '--xcore_tuned', action='store_true', default=False,
        help='Use a variation of the model tuned for xcore.ai.')
    args = parser.parse_args()

    utils.set_verbosity(args.verbose)
    utils.set_gpu_usage(args.use_gpu, args.verbose)

    main(path=args.path,
         train_new_model=args.train_model,
         batch_size=args.batch,
         epochs=args.epochs,
         use_aug=args.augment_dataset,
         xcore_tuned=args.xcore_tuned,
         opt_classifier=args.classifier)
