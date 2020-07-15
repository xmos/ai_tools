#!/usr/bin/env python
#
# Copyright (c) 2020, XMOS Ltd, All rights reserved

from pathlib import Path
from imagenet_common import ImagenetteModel, DefaultParser
import tensorflow as tf
import numpy as np

DEFAULT_NAME = "mobilenet"
DEFAULT_PATH = Path(__file__).parent.joinpath("debug", DEFAULT_NAME).resolve()
DEFAULT_NUM_THREADS = 1


class Mobilenet(ImagenetteModel):
    def build(self, **kwargs):
        assert "classes" not in kwargs, "classes should be set in constructor"
        assert (
            "input_shape" not in kwargs
        ), "input_shape should be set in constructor via input_size"

        # the new input_shape is only used to fetch weights
        canonical_sizes = [128, 160, 192, 224]
        max_input_size = max(self.input_size)
        ref_idx = np.abs(np.array(canonical_sizes) - max_input_size).argmin()
        ref_input_size = canonical_sizes[ref_idx]
        kwargs["input_shape"] = (ref_input_size, ref_input_size, 3)

        # the actual input shape is enforced by passing an explicit input layer
        kwargs.setdefault(
            "input_tensor", tf.keras.layers.Input(shape=(*self.input_size, 3))
        )

        self._prep_backend()

        include_top = kwargs.pop("include_top", True)
        if not include_top:
            self.core_model = tf.keras.applications.MobileNet(
                include_top=False, **kwargs
            )
            self.core_model.summary()
            return

        # load base_model for conv weights, source model for FC weights
        base_model = tf.keras.applications.MobileNet(include_top=False, **kwargs)
        source_model = tf.keras.applications.MobileNet(**kwargs)

        # extract weights of dense layer from source model
        w, b = source_model.layers[90].get_weights()
        assert b.shape[0] == w.shape[-1] == 1000
        wt, bt = w[0, 0, :, self.classes].T, b[self.classes]

        # add new top (i.e. classifier)
        self.core_model = tf.keras.Sequential(
            [
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(
                    len(self.classes),
                    activation="softmax",
                    kernel_initializer=tf.keras.initializers.Constant(wt.tolist()),
                    bias_initializer=tf.keras.initializers.Constant(bt.tolist()),
                ),
            ]
        )
        self.core_model.build(self.input_shape)

        # Quick check, are we adding the right weights in the right place?
        wn, bn = self.core_model.layers[2].get_weights()
        assert np.all(wn == wt) and np.all(bn == bt)

        self.core_model.summary()


def main(raw_args=None):
    parser = DefaultParser(defaults={"path": DEFAULT_PATH})
    parser.add_argument(
        "-i",
        "--input_size",
        nargs="+",
        type=int,
        default=[128, 128],
        help="Input image size in pixels [height, [width]]. "
        "If only one number is specified it will be used for both dimensions.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        choices=[0.25, 0.5, 0.75, 1.0],
        help="Alpha parameter for MobileNet.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=[10],
        help="Target classes. If a single number 'n' is given, the first 'n' "
        "Imagenette classes are targeted. If a list of integers is given, "
        "the classes with the given indices as targeted",
    )
    parser.add_argument(
        "-par",
        "--num_threads",
        type=int,
        default=DEFAULT_NUM_THREADS,
        help="Number of parallel threads for xcore.ai optimization.",
    )
    args = parser.parse_args(raw_args)

    model = Mobilenet(
        name=DEFAULT_NAME,
        path=args.path,
        classes=args.classes,
        input_size=args.input_size,
    )
    model.build(alpha=args.alpha)
    model.gen_test_data()
    model.save_core_model()
    model.convert_and_save(xcore_num_threads=args.num_threads)


if __name__ == "__main__":
    main()
