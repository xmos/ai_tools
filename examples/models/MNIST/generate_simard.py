#!/usr/bin/env python
#
# Copyright (c) 2020, XMOS Ltd, All rights reserved
from pathlib import Path
from tflite2xcore.model_generation import utils
from mnist_common import MNISTModel, get_default_parser, run_main
import tensorflow as tf

DEFAULT_PATH = {
    'simard': Path(__file__).parent.joinpath('debug', 'simard').resolve(),
    'simard_tuned': Path(__file__).parent.joinpath('debug', 'simard_tuned').resolve(),
    'simard_cls': Path(__file__).parent.joinpath('debug', 'simard_cls').resolve()
}
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


def main(path=None, train_new_model=False,
         batch_size=DEFAULT_BS, epochs=DEFAULT_EPOCHS,
         use_aug=False, xcore_tuned=False, opt_classifier=False):

    name = ('simard_cls' if opt_classifier else 'simard_tuned') if xcore_tuned else 'simard'
    kwargs = {
        'name': name,
        'path': path if path else DEFAULT_PATH[name],
        'opt_classifier': opt_classifier,
        'use_aug': use_aug
    }

    run_main(
        model=SimardTuned(**kwargs) if xcore_tuned else Simard(**kwargs),
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
