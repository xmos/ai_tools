#!/usr/bin/env python
# Copyright 2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Type, NamedTuple, Union

from tflite2xcore import utils
from tflite2xcore.model_generation import ModelGenerator, Configuration
from tflite2xcore.model_generation.data_factories import DataFactory
from tflite2xcore.model_generation.runners import Runner
from tflite2xcore.model_generation.converters import (
    TFLiteQuantConverter,
    XCoreConverter,
)


class TrainingData(NamedTuple):
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


class NormalizedCIFAR10Factory(DataFactory):
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


class TrainableModelGenerator(ModelGenerator):
    _runner: "TrainingRunner"

    def _set_config(self, cfg: Configuration) -> None:
        self._config["epochs"] = cfg.pop("epochs")
        self._config["batch_size"] = cfg.pop("batch_size")
        super()._set_config(cfg)

    def train(self) -> None:
        data = self._runner.get_training_data()
        self._model.fit(
            data.x_train,
            data.y_train,
            validation_data=(data.x_test, data.y_test),
            batch_size=self._config["batch_size"],
            epochs=self._config["epochs"],
        )


class TrainingRunner(Runner):
    _data: TrainingData
    _model_generator: TrainableModelGenerator

    def __init__(self, generator: Type[TrainableModelGenerator]) -> None:
        self._data_factory = NormalizedCIFAR10Factory(self)

        self._quant_converter = TFLiteQuantConverter(
            self, self._model_generator.get_model, self.get_quantization_data
        )
        self._xcore_converter = XCoreConverter(self, self._model_generator.get_model)
        super().__init__(
            generator,
            converters=[self._quant_converter, self._xcore_converter],
            data_factories=[self._data_factory],
        )

    def run(self) -> None:
        super().run()
        self._model_generator.train()
        for converter in self._converters:
            converter.convert()
        self.converted_models.update(
            {
                "quant": self._quant_converter._model,
                "xcore": self._xcore_converter._model,
            }
        )

    def get_training_data(self) -> TrainingData:
        try:
            return self._data
        except AttributeError:
            self._data = self._data_factory.make_data()
            return self._data

    def get_quantization_data(self) -> tf.Tensor:
        data = self.get_training_data()

        sorted_inds = np.argsort(data.y_test, axis=0, kind="mergesort")
        subset_inds = np.searchsorted(data.y_test[sorted_inds].flatten(), np.arange(10))
        subset_inds = sorted_inds[subset_inds]
        return data.x_test[subset_inds.flatten()]


class ArmBenchmarkModelGenerator(TrainableModelGenerator):
    def build(self) -> None:
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


if __name__ == "__main__":
    parser = utils.VerbosityParser()
    parser.add_argument(
        "-d",
        "--model_dir",
        required=False,
        default=None,
        help="Directory for converted models.",
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    args = parser.parse_args()

    utils.set_verbosity(args.verbose)
    model_dir = args.model_dir
    model_dir = Path(model_dir) if model_dir else Path(__file__).parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    runner = TrainingRunner(ArmBenchmarkModelGenerator)
    runner.set_config(epochs=args.epochs, batch_size=args.batch_size)
    runner.run()
    runner.dump_models(model_dir.resolve())
