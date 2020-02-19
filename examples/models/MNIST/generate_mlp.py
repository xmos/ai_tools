#!/usr/bin/env python
#
# Copyright (c) 2020, XMOS Ltd, All rights reserved
from pathlib import Path
from tflite2xcore.model_generation import utils
from mnist_common import MNISTModel, get_default_parser, run_main
import tensorflow as tf

DEFAULT_PATH = {
    'mlp': Path(__file__).parent.joinpath('debug', 'mlp').resolve(),
    'mlp_tuned': Path(__file__).parent.joinpath('debug', 'mlp_tuned').resolve(),
    'mlp_cls': Path(__file__).parent.joinpath('debug', 'mlp_cls').resolve()
}
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


def main(path=None, train_new_model=False,
         batch_size=DEFAULT_BS, epochs=DEFAULT_EPOCHS,
         use_aug=False, xcore_tuned=False, opt_classifier=False):

    name = ('mlp_cls' if opt_classifier else 'mlp_tuned') if xcore_tuned else 'mlp'
    kwargs = {
        'name': name,
        'path': path if path else DEFAULT_PATH[name],
        'opt_classifier': opt_classifier,
        'use_aug': use_aug
    }

    run_main(
        model=MLPTuned(**kwargs) if xcore_tuned else MLP(**kwargs),
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
