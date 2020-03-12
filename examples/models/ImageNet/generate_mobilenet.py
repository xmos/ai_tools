#!/usr/bin/env python
#
# Copyright (c) 2020, XMOS Ltd, All rights reserved
from pathlib import Path
from imagenet_common import ImageNetModel, ImagenetDefaultParser
import tensorflow as tf
import numpy as np
from mobilenet_builder import MobileNet


DEFAULT_NAME = 'mobilenet'
DEFAULT_PATH = Path(__file__).parent.joinpath('debug', DEFAULT_NAME).resolve()
DEFAULT_EPOCHS = 10
DEFAULT_BS = 64


class Mobilenet(ImageNetModel):
    def build(self, num_classes=100):
        self._prep_backend()
        # Base model without classifier or padding in between the depthwise convs
        base_model = MobileNet(include_top=False,
                               depth_multiplier=1, weights='imagenet',
                               input_shape=(159, 159, 3), alpha=0.25)

        # model to which extract the weights of the last layer
        top_weights = tf.keras.applications.MobileNet(include_top=True,
                                                      depth_multiplier=1,
                                                      weights='imagenet',
                                                      input_shape=(128, 128, 3),
                                                      alpha=0.25)
        top_weights.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.RMSprop(),
            metrics=['accuracy'])
        w, b = top_weights.layers[90].get_weights()
        wt, bt = w[0, 0, :, :num_classes], b[:num_classes]
        global_avg_layer = tf.keras.layers.GlobalAveragePooling2D()
        preds = tf.keras.layers.Dense(num_classes, activation='softmax',
                                      kernel_initializer=tf.keras.initializers.Constant(wt.tolist()),
                                      bias_initializer=tf.keras.initializers.Constant(bt.tolist()))
        self.core_model = tf.keras.Sequential([
            base_model, global_avg_layer, preds])
        self.core_model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.RMSprop(),
            metrics=['accuracy']
        )
        # Quick check, are we adding the right weights in the right place?
        wn, bn = self.core_model.layers[2].get_weights()
        assert np.all(wn == wt) and np.all(bn == bt)
        self.core_model.summary()


def main(raw_args=None):
    parser = ImagenetDefaultParser(defaults={
        'path': DEFAULT_PATH,
    })
    args = parser.parse_args()
    model = Mobilenet(name=DEFAULT_NAME, path=args.path)
    model.run()


if __name__ == "__main__":
    main()
