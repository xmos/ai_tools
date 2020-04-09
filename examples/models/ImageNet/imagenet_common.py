# Copyright (c) 2020, XMOS Ltd, All rights reserved

# TODO: fix this hack
from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(realpath(__file__))))

from model_common import DefaultParser

from pathlib import Path
from tflite2xcore.model_generation import utils
from tflite2xcore.model_generation.interface import KerasClassifier
import tensorflow as tf
import numpy as np


# TODO: fix this
KOALA_urls = [
    "http://farm1.static.flickr.com/159/403176078_a2415ddf33.jpg",
    "http://farm1.static.flickr.com/179/423878571_29ce38383e.jpg",
    "http://farm2.static.flickr.com/1330/1135686352_43553d0dac.jpg",
    "http://farm1.static.flickr.com/136/378225968_28eb9274cd.jpg",
    "http://farm1.static.flickr.com/154/415086724_ceb3964c77.jpg",
    "http://farm3.static.flickr.com/2195/2079170850_5952195903.jpg",
    "http://farm1.static.flickr.com/157/399669613_8180eb8e83.jpg",
    "http://farm2.static.flickr.com/1208/558914192_f0302b27f0.jpg",
    "http://static.flickr.com/31/382204367_abfc8cc74a.jpg",
    "http://farm2.static.flickr.com/1365/867539333_6b17578bbd.jpg",
]


class ImageNetModel(KerasClassifier):
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

    def gen_test_data(self, target_size=None):
        target_size = target_size or (128, 128)

        # TODO: fix this
        koalas = []
        for j, url in enumerate(KOALA_urls):
            f = tf.keras.utils.get_file(f"koala_{j}.jpg", url)
            img = tf.keras.preprocessing.image.load_img(f, target_size=target_size)
            x = tf.keras.preprocessing.image.img_to_array(img).astype(np.float32)
            koalas.append(x)
        example_tensor = (np.stack(koalas) / 127.5 - 1.)
        self.data['export'] = example_tensor
        self.data['quant'] = example_tensor
