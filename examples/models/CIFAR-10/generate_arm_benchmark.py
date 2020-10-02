#!/usr/bin/env python
#
# Copyright (c) 2020, XMOS Ltd, All rights reserved

import logging
import numpy as np
import tensorflow as tf
from abc import abstractmethod
from typing import Type, NamedTuple, Union

DEFAULT_EPOCHS = 30
DEFAULT_BS = 32


from tflite2xcore.model_generation import ModelGenerator
from tflite2xcore.model_generation.data_factories import DataFactory
from tflite2xcore.model_generation.runners import Runner


class TrainingData(NamedTuple):
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


class TrainingDataFactory(DataFactory):
    @abstractmethod
    def make_data(self) -> TrainingData:
        raise NotImplementedError()


class NormalizedCIFAR10Factory(TrainingDataFactory):
    @staticmethod
    def normalize(data: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
        scale = tf.constant(255, dtype=tf.dtypes.float32)
        return data / scale - 0.5

    def make_data(self) -> TrainingData:
        (
            (train_images, train_labels),
            (test_images, test_labels),
        ) = tf.keras.datasets.cifar10.load_data()

        return TrainingData(
            np.float32(self.normalize(train_images)),
            np.float32(train_labels),
            np.float32(self.normalize(test_images)),
            np.float32(test_labels),
        )


class TrainingRunner(Runner):
    def __init__(self, generator: Type[ModelGenerator]) -> None:
        super().__init__(generator)

    def run(self):
        super().run()
        data = NormalizedCIFAR10Factory(self).make_data()
        self._model_generator.train(data)


class TrainableModelGenerator(ModelGenerator):
    def train(self, data: TrainingData):
        fit_args = dict(
            validation_data=(data.x_test, data.y_test), epochs=DEFAULT_EPOCHS,
        )
        fit_args["x"] = data.x_train
        fit_args["y"] = data.y_train
        fit_args["batch_size"] = DEFAULT_BS

        self._model.fit(**fit_args)


class ArmBenchmarkModelGenerator(TrainableModelGenerator):
    def build(self):
        self._prep_backend()
        self._model = tf.keras.Sequential(
            layers=[
                tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
                tf.keras.layers.experimental.preprocessing.RandomFlip(
                    mode="horizontal"
                ),
                tf.keras.layers.experimental.preprocessing.RandomTranslation(
                    0.1, 0.1, fill_mode="reflect", interpolation="nearest",
                ),
                tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(filters=64, kernel_size=5, padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(10, activation="softmax"),
            ],
        )

        self._model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )
        self._model.summary()

    # def gen_test_data(self):
    #     if not self.data:
    #         self.prep_data()

    #     sorted_inds = np.argsort(self.data["y_test"], axis=0, kind="mergesort")
    #     subset_inds = np.searchsorted(
    #         self.data["y_test"][
    #             sorted_inds
    #         ].flatten(),  # pylint: disable=unsubscriptable-object
    #         np.arange(10),
    #     )
    #     subset_inds = sorted_inds[subset_inds]
    #     self.data["export"] = self.data["x_test"][
    #         subset_inds.flatten()
    #     ]  # pylint: disable=unsubscriptable-object
    #     self.data["quant"] = self.data["x_train"]


if __name__ == "__main__":
    runner = TrainingRunner(ArmBenchmarkModelGenerator)
    runner.run()
