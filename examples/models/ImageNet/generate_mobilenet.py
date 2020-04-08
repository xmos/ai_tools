#!/usr/bin/env python
#
# Copyright (c) 2020, XMOS Ltd, All rights reserved

from pathlib import Path
from imagenet_common import ImageNetModel, DefaultParser
import tensorflow as tf
import numpy as np

DEFAULT_NAME = 'mobilenet'
DEFAULT_PATH = Path(__file__).parent.joinpath('debug', DEFAULT_NAME).resolve()


class Mobilenet(ImageNetModel):
    def build(self, **kwargs):
        include_top = kwargs.pop('include_top', True)
        classes = kwargs.pop('classes', 1000)
        if isinstance(classes, list):
            classes = np.unique(classes).tolist()
            assert 0 < len(classes) <= 1000
            assert classes[0] >= 0
            assert classes[-1] < 1000
        else:
            classes = list(range(classes))

        self._prep_backend()

        if not include_top:
            self.core_model = tf.keras.applications.MobileNet(**kwargs)
            self.core_model.summary()
            return

        base_model = tf.keras.applications.MobileNet(include_top=False, **kwargs)

        input_shape = kwargs.pop('input_shape')
        if (input_shape[0] != input_shape[1]
                or input_shape[0] not in [128, 160, 192, 224]):
            input_shape = (224, 224, 3)

        source_model = tf.keras.applications.MobileNet(
            classes=1000, input_shape=input_shape, **kwargs
        )

        # extract weights of dense layer from source model
        w, b = source_model.layers[90].get_weights()
        assert b.shape[0] == w.shape[-1] == 1000
        wt, bt = w[0, 0, :, classes].T, b[classes]

        # add new classifier
        self.core_model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(
                len(classes),
                activation='softmax',
                kernel_initializer=tf.keras.initializers.Constant(wt.tolist()),
                bias_initializer=tf.keras.initializers.Constant(bt.tolist()))
        ])
        self.core_model.build(input_shape)

        # Quick check, are we adding the right weights in the right place?
        wn, bn = self.core_model.layers[2].get_weights()
        assert np.all(wn == wt) and np.all(bn == bt)
        self.core_model.summary()


def main(raw_args=None):
    parser = DefaultParser(defaults={
        'path': DEFAULT_PATH,
    })
    parser.add_argument(
        '--classifier', action='store_true', default=False,
        help='Apply classifier optimizations during xcore conversion.'
    )
    parser.add_argument(
        '-h', '--height', type=int, default=128,
        help='Image height in pixels.'
    )
    parser.add_argument(
        '-w', '--width', type=int, default=128,
        help='Image width in pixels.'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.25, choices=[0.25, 0.5, 0.75, 1.0],
        help='Alpha parameter for MobileNet.'
    )
    parser.add_argument(
        '--classes', nargs="+", type=int, default=150,
        help="Target classes. If a single number 'n' is given, the first 'n' "
             "ImageNet classes are targeted. If a list of inteegers is given, "
             "the classes with the given indices as targeted"
    )
    args = parser.parse_args(raw_args)

    model = Mobilenet(
        name=DEFAULT_NAME, path=args.path, opt_classifier=args.classifier)
    model.build(input_shape=(args.height, args.width, 3),
                classes=args.classes,
                alpha=args.alpha)
    model.gen_test_data(target_size=(args.height, args.width))
    model.save_core_model()
    model.convert_and_save()


if __name__ == "__main__":
    main()
