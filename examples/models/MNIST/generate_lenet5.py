#!/usr/bin/env python
# Copyright 2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
from pathlib import Path
from mnist_common import MNISTModel, XcoreTunedParser
import tensorflow as tf
from tensorflow.keras import layers

DEFAULT_PATH = Path(__file__).parent.joinpath("debug")
DEFAULT_NAME = "lenet5"
DEFAULT_EPOCHS = 10
DEFAULT_BS = 64


class LeNet5(MNISTModel):
    def build(self):
        self._prep_backend()
        # Building
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.Input(shape=(32, 32, 1), name="input"),
                layers.Conv2D(6, kernel_size=5, name="conv_1"),
                layers.BatchNormalization(),
                layers.Activation("tanh"),
                layers.AvgPool2D(pool_size=2, strides=2, name="avg_pool_1"),
                layers.Conv2D(16, kernel_size=5, name="conv_2"),
                layers.BatchNormalization(),
                layers.Activation("tanh"),
                layers.AvgPool2D(pool_size=2, strides=2, name="avg_pool_2"),
                layers.Conv2D(120, kernel_size=5, name="conv_3"),
                layers.Activation("tanh"),
                layers.Flatten(),
                layers.Dense(84, activation="tanh", name="fc_1"),
                layers.Dense(10, activation="softmax", name="output"),
            ],
        )
        opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-2 / 10)
        # Compilation
        self.core_model.compile(
            loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
        )
        # Show summary
        self.core_model.summary()

    def train(self, *, batch_size, save_history=True, **kwargs):
        # Image generator, # TODO: make this be optional with self._use_aug
        aug = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest",
        )
        # Train the network
        self.history = self.core_model.fit_generator(
            aug.flow(self.data["x_train"], self.data["y_train"], batch_size=batch_size),
            validation_data=(self.data["x_test"], self.data["y_test"]),
            steps_per_epoch=len(self.data["x_train"]) // batch_size,
            **kwargs
        )
        if save_history:
            self.save_training_history()


class LeNet5Tuned(LeNet5):
    def build(self):
        self._prep_backend()
        # Building
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.Input(shape=(32, 32, 1), name="input"),
                layers.Conv2D(8, kernel_size=5, name="conv_1"),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.AvgPool2D(pool_size=2, strides=2, name="avg_pool_1"),
                layers.Conv2D(16, kernel_size=5, name="conv_2"),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.AvgPool2D(pool_size=2, strides=2, name="avg_pool_2"),
                layers.Conv2D(128, kernel_size=5, name="conv_3"),
                layers.ReLU(),
                layers.Flatten(),
                layers.Dense(96, activation="relu", name="fc_1"),
                layers.Dense(10, activation="softmax", name="output"),
            ],
        )
        opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-2 / 10)
        # 10 epochs with categorical data
        # Compilation
        self.core_model.compile(
            loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
        )
        # Show summary
        self.core_model.summary()


def main(raw_args=None):
    parser = XcoreTunedParser(
        defaults={
            "batch_size": DEFAULT_BS,
            "epochs": DEFAULT_EPOCHS,
            "name": DEFAULT_NAME,
            "path": DEFAULT_PATH,
        }
    )
    args = parser.parse_args(raw_args)

    kwargs = {"name": args.name, "path": args.path, "use_aug": args.augment_dataset}
    model = LeNet5Tuned(**kwargs) if args.xcore_tuned else LeNet5(**kwargs)
    model.run(
        train_new_model=args.train_model, batch_size=args.batch_size, epochs=args.epochs
    )


if __name__ == "__main__":
    main()
