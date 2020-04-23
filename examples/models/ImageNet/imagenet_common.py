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
import numpy as np
import json


with open(Path(__file__).parent / "example_urls.json", 'r') as f:
    IMAGENET_URLS = json.load(f)

CLASS_INDEX_PATH = ("https://storage.googleapis.com/download.tensorflow.org"
                    "/data/imagenet_class_index.json")

fpath = tf.keras.utils.get_file(
    'imagenet_class_index.json',
    CLASS_INDEX_PATH,
    cache_subdir='models',
    file_hash='c2c37ea517e94d9795004a39431a14cb'
)
with open(fpath) as f:
    CLASS_INDEX = json.load(f)


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

    def gen_test_data(self, *, samples_per_class=10):
        cache_dir = self._path / "cache" / "imagenet"
        cache_dir.mkdir(exist_ok=True, parents=True)

        examples = []
        for class_idx in self.classes:
            class_key = str(class_idx)
            class_name = CLASS_INDEX[class_key][1]
            class_urls = IMAGENET_URLS[class_key][:samples_per_class]
            assert len(class_urls) == samples_per_class
            print(f"Loading images for class {class_idx}: '{class_name}'...")

            class_examples = []
            for j, url in enumerate(class_urls):
                f = tf.keras.utils.get_file(
                    f"{j:02d}.jpg",
                    origin=url,
                    cache_dir=cache_dir,
                    cache_subdir="_".join([f"{class_idx:03d}", class_name])
                )
                img = tf.keras.preprocessing.image.load_img(
                    f, target_size=self.input_size
                )
                class_examples.append(
                    tf.keras.preprocessing.image.img_to_array(img).astype(np.float32)
                )
            assert len(class_examples) == samples_per_class

            examples += class_examples

        examples = (np.stack(examples) / 127.5 - 1.)
        self.data['export'] = examples
        self.data['quant'] = examples
