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

        # the actual input shape is enforced by passing an explicit input layer
        input_shape = kwargs['input_shape']
        kwargs.setdefault('input_tensor', tf.keras.layers.Input(input_shape))

        # the new input_shape is only used to fetch weights
        canonical_sizes = [128, 160, 192, 224]
        input_size = max(input_shape[:2])
        ref_idx = np.abs(np.array(canonical_sizes) - input_size).argmin()
        ref_input_size = canonical_sizes[ref_idx]
        kwargs['input_shape'] = (ref_input_size, ref_input_size, 3)

        self._prep_backend()

        if not include_top:
            self.core_model = tf.keras.applications.MobileNet(**kwargs)
            self.core_model.summary()
            return

        # load base_model for conv weights, source model for FC weights
        base_model = tf.keras.applications.MobileNet(include_top=False, **kwargs)
        source_model = tf.keras.applications.MobileNet(**kwargs)

        # extract weights of dense layer from source model
        w, b = source_model.layers[90].get_weights()
        assert b.shape[0] == w.shape[-1] == 1000
        wt, bt = w[0, 0, :, classes].T, b[classes]

        # add new top (i.e. classifier)
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
        '-i', '--input_size', nargs="+", type=int, default=[128, 128],
        help='Input image size in pixels [height, width].'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.25, choices=[0.25, 0.5, 0.75, 1.0],
        help='Alpha parameter for MobileNet.'
    )
    parser.add_argument(
        '--classes', nargs="+", type=int, default=10,
        help="Target classes. If a single number 'n' is given, the first 'n' "
             "ImageNet classes are targeted. If a list of inteegers is given, "
             "the classes with the given indices as targeted"
    )
    args = parser.parse_args(raw_args)

    if len(args.input_size) > 2:
        raise ValueError("Input image size must be one or two numbers!")
    elif len(args.input_size) == 1:
        args.input_size *= 2

    model = Mobilenet(
        name=DEFAULT_NAME, path=args.path, opt_classifier=args.classifier)
    model.build(input_shape=(*args.input_size, 3),
                classes=args.classes,
                alpha=args.alpha)
    model.gen_test_data(target_size=args.input_size)
    model.save_core_model()
    model.convert_and_save()


if __name__ == "__main__":
    main()
