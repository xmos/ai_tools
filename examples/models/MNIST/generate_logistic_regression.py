#!/usr/bin/env python
#
# Copyright (c) 2020, XMOS Ltd, All rights reserved
from pathlib import Path
from tflite2xcore.model_generation import utils
from mnist_common import MNISTModel, get_default_parser, run_main
import tensorflow as tf

DEFAULT_PATH = {
    'logistic_regression': Path(__file__).parent.joinpath('debug', 'logistic_regression').resolve(),
    'logistic_regression_cls': Path(__file__).parent.joinpath('debug', 'logistic_regression_cls').resolve()
}
DEFAULT_EPOCHS = 10
DEFAULT_BS = 512


class LogisticRegression(MNISTModel):

    def build(self):
        self._prep_backend()
        utils.set_all_seeds(seed=42)
        # Building
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.layers.Flatten(input_shape=(28, 28, 1), name='input'),
                tf.keras.layers.Dense(10, activation='softmax',
                                      kernel_regularizer=tf.keras.regularizers.l1(0.00019899999999999999))
            ]
        )
        # Compilation
        self.core_model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(0.012877456566121567),
            metrics=['accuracy']
        )
        # Show summary
        self.core_model.summary()

    def prep_data(self):
        super().prep_data(padding=0)


def main(path=None, train_new_model=False,
         batch_size=DEFAULT_BS, epochs=DEFAULT_EPOCHS,
         use_aug=False, opt_classifier=False):
    name = 'logistic_regression_cls' if opt_classifier else 'logistic_regression'
    kwargs = {
        'name': name,
        'path': path if path else DEFAULT_PATH[name],
        'opt_classifier': opt_classifier,
        'use_aug': use_aug
    }
    run_main(
        model=LogisticRegression(**kwargs),
        train_new_model=train_new_model,
        batch_size=batch_size, epochs=epochs
    )


if __name__ == '__main__':
    parser = get_default_parser(DEFAULT_BS=DEFAULT_BS, DEFAULT_EPOCHS=DEFAULT_EPOCHS)
    args = parser.parse_args()

    utils.set_verbosity(args.verbose)
    utils.set_gpu_usage(args.use_gpu, args.verbose)

    main(path=args.path,
         train_new_model=args.train_model,
         batch_size=args.batch,
         epochs=args.epochs,
         use_aug=args.augment_dataset,
         opt_classifier=args.classifier)
