# Copyright (c) 2020, XMOS Ltd, All rights reserved

# TODO: fix this hack
from os.path import dirname, realpath
import sys

sys.path.append(dirname(dirname(realpath(__file__))))

from model_common import DefaultParser

from pathlib import Path
from tflite2xcore.model_generation import utils
from tflite2xcore.model_generation.interface import KerasModel
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import json
import itertools


with open(Path(__file__).parent / "example_urls.json", "r") as f:
    IMAGENET_URLS = json.load(f)

CLASS_INDEX_PATH = (
    "https://storage.googleapis.com/download.tensorflow.org"
    "/data/imagenet_class_index.json"
)

fpath = tf.keras.utils.get_file(
    "imagenet_class_index.json",
    CLASS_INDEX_PATH,
    cache_subdir="models",
    file_hash="c2c37ea517e94d9795004a39431a14cb",
)
with open(fpath) as f:
    IMAGENET_CLASS_INDEX = json.load(f)

IMAGENETTE_CLASSES = [
    "tench",
    "English_springer",
    "cassette_player",
    "chain_saw",
    "church",
    "French_horn",
    "garbage_truck",
    "gas_pump",
    "golf_ball",
    "parachute",
]

IMAGENETTE_CLASS_INDEX = {
    idx: v for idx, v in IMAGENET_CLASS_INDEX.items() if v[1] in IMAGENETTE_CLASSES
}

IMAGENETTE_DS = tfds.load("imagenette/320px-v2", split="validation")


class ImageNetModel(KerasModel):
    def __init__(self, *args, classes=None, input_size=None, **kwargs):
        super().__init__(*args, **kwargs)

        classes = classes or 1000
        if isinstance(classes, int):
            classes = [classes]
        if isinstance(classes, (tuple, list)):
            if len(classes) == 1:
                self.classes = list(range(classes[0]))
            else:
                self.classes = np.unique(classes).tolist()
        else:
            raise TypeError("classes must be an integer or tuple/list.")
        assert 1 < len(self.classes) <= 1000
        assert self.classes[0] >= 0
        assert self.classes[-1] < 1000

        input_size = input_size or 128
        if isinstance(input_size, int):
            input_size = (input_size,)
        if isinstance(input_size, (tuple, list)):
            if len(input_size) > 2:
                raise ValueError("input_size must be one or two numbers!")
            elif len(input_size) == 1:
                input_size *= 2
        else:
            raise TypeError("input_size must be an integer or tuple/list.")
        self.input_size = input_size

    def train(self):
        pass

    def prep_data(self):
        pass


class ImagenetteModel(ImageNetModel):
    def gen_test_data(self, *, samples_per_class=10):
        assert len(self.classes) <= 10, "At most 10 classes can be used with Imagenette"
        assert all(
            class_idx < 10 for class_idx in self.classes
        ), "Imagenette indexes range from 0 through 9"

        print("Loading images...")
        examples = {class_idx: [] for class_idx in self.classes}
        for d in IMAGENETTE_DS:
            class_idx = int(d["label"])
            if class_idx in self.classes:
                if len(examples[class_idx]) < samples_per_class:
                    examples[class_idx].append(
                        tf.image.resize(
                            d["image"].numpy().astype(np.float32) / 127.5 - 1.0,
                            self.input_size,
                        )
                    )

        examples = np.stack(list(itertools.chain(*examples.values())))

        self.data["export"] = examples
        self.data["quant"] = examples
