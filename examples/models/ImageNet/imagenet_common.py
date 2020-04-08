# Copyright (c) 2020, XMOS Ltd, All rights reserved

# TODO: fix this hack
from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(realpath(__file__))))

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
    def train(self):
        pass

    def prep_data(self):
        pass

    def gen_test_data(self, target_size=(128, 128)):
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
