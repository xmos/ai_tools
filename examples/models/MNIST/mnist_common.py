# Copyright (c) 2020, XMOS Ltd, All rights reserved

# TODO: fix this hack
from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(realpath(__file__))))

from model_common import TrainableParser

import numpy as np

from pathlib import Path
from tqdm import tqdm
from tflite2xcore.model_generation import utils
from tflite2xcore.model_generation.interface import KerasModel
from tflite2xcore import xlogging as logging

import tensorflow as tf
from sklearn.utils import shuffle
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


#  ----------------------------------------------------------------------------
#                                  UTILITIES
#  ----------------------------------------------------------------------------

def get_mnist(padding=2, categorical=False, val_split=True, flatten=False,
              debug=True, y_float=False):
    '''
    Get the keras MNIST dataset in the specified format.
    \t- categorical: if categorical labels or not
    \t- padding: if padding of the images or not
    \t- val_split: if divide into validation as well or not
    \t- flatten: if we want the output datasets to have only 2 dims or not
    \t- debug: if we want printed shapes and extra information or not
    \t- y_float: if we want the labels to be float numbers
    '''
    rows = 28
    cols = 28
    nb_classes = 10

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(
        x_train.shape[0], rows, cols, 1).astype('float32')/255
    x_test = x_test.reshape(
        x_test.shape[0], rows, cols, 1).astype('float32')/255

    if y_float:
        y_train = y_train.astype('float32')
        y_test = y_test.astype('float32')

    if padding:
        x_train = np.pad(x_train,
                         ((0, 0), (padding, padding),
                          (padding, padding), (0, 0)), 'constant')
        x_test = np.pad(x_test,
                        ((0, 0), (padding, padding),
                         (padding, padding), (0, 0)), 'constant')

    if categorical:
        y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
        y_test = tf.keras.utils.to_categorical(y_test, nb_classes)
        y_train = y_train.reshape(y_train.shape[0], 10)
        y_test = y_test.reshape(y_test.shape[0], 10)

    if val_split:
        index = int(0.8 * len(x_train))
        x_train, x_val = x_train[:index], x_train[index:]
        y_train, y_val = y_train[:index], y_train[index:]

    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
        if val_split:
            x_val = x_val.reshape(x_val.shape[0], -1)
    if not categorical:
        train_labels_count = np.unique(y_train, return_counts=True)
        logging.getLogger().debug(f"labels counts: {train_labels_count[1]}")
    if val_split:
        return x_train, x_test, x_val, y_train, y_test, y_val
    return x_train, x_test, y_train, y_test


# TODO: change name to something more meaningful
def ecc(nsizex=29, nsizey=29, ch=1):
    '''
    Crop the dataset images using resize from skimage,
    consider instead use keras layer Cropping2D.
    '''
    x_train, x_test, x_val, y_train, y_test, y_val = get_mnist(
        padding=0, categorical=False)
    import skimage.transform
    with tqdm(total=30) as pbar:
        o_train = skimage.transform.resize(x_train, (x_train.shape[0], nsizex, nsizey, ch))
        pbar.update(10)
        o_test = skimage.transform.resize(x_test, (x_test.shape[0], nsizex, nsizey, ch))
        pbar.update(10)
        o_val = skimage.transform.resize(x_val, (x_val.shape[0], nsizex, nsizey, ch))
        pbar.update(10)
    return o_train, o_test, o_val, y_train, y_test, y_val


def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape) == 2
    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(shape)


# Apply elastic distortions to the input
# images: set of images; labels: associated labels
def expand_dataset(images, labels, distortions, sigma=4.0, alpha=60.0,
                   sizex=32, sizey=32):
    '''
    Function to expand a dataset with more images.
    \t- images: original dataset (numpy array)
    \t- labels: original dataset labels (numpy array)
    \t- distortions: number of distortions per image
    \t- sigma: sigma value
    \t- alpha: alpha value
    \t- sizex: size x of the image
    \t- sizey: size y of the image
    '''
    new_images_batch = np.array(
        [elastic_transform(np.reshape(image, (sizex, sizey)), alpha, sigma)
         for image in tqdm(images) for _ in range(distortions)])
    new_labels_batch = np.array(
        [label for label in tqdm(labels) for _ in range(distortions)])
    # Don't forget to return the original images and labels (hence concatenate)
    x_data = np.concatenate([np.reshape(images, (-1, sizex, sizey)), new_images_batch])
    y_data = np.concatenate([labels, new_labels_batch])
    return x_data.reshape(x_data.shape[0], sizex, sizey, 1), y_data


def prepare_MNIST(use_aug=False, simard=False, padding=2):
    if simard:
        x_train, x_test, x_val, y_train, y_test, y_val = ecc()
    else:
        x_train, x_test, x_val, y_train, y_test, y_val = get_mnist(
            padding=padding, categorical=False, flatten=False, y_float=True)
    if use_aug:
        if simard:
            x_train, y_train = expand_dataset(
                x_train, y_train, 2, sigma=4.0, alpha=16.0,
                sizex=29, sizey=29)
        else:
            x_train, y_train = expand_dataset(
                x_train, y_train, 2, sigma=4.0, alpha=16.0)
    x_train, y_train = shuffle(x_train, y_train)

    return {'x_train': np.float32(x_train[:4096]),
            'x_test': np.float32(x_test[:1024]),
            'x_val': np.float32(x_val[:100]),
            'y_train': np.float32(y_train[:4096]),
            'y_test': np.float32(y_test[:1024]),
            'y_val': np.float32(y_val[:100])}


#  ----------------------------------------------------------------------------
#                                  MODELS
#  ----------------------------------------------------------------------------

class MNISTModel(KerasModel):
    def __init__(self, *args, use_aug=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_aug = use_aug

    def prep_data(self, *, simard_resize=False, padding=2):
        self.data = prepare_MNIST(self._use_aug, simard=simard_resize, padding=padding)
        for k, v in self.data.items():
            self.logger.debug(f"Prepped data[{k}] with shape: {v.shape}")

    def gen_test_data(self):
        if not self.data:
            self.prep_data()
        self.data['export'] = self.data['x_test'][:10]
        self.data['quant'] = self.data['x_train'][:10]

    def run(self, train_new_model=False, epochs=None, batch_size=None):
        self._prep_backend()
        if train_new_model:
            assert epochs
            assert batch_size
            # Build model and compile
            self.build()
            # Prepare training data
            self.prep_data()
            # Train model
            self.train(batch_size=batch_size, epochs=epochs)
            self.save_core_model()
        else:
            # Recover previous state from file system
            self.load_core_model()
        self.convert_and_save()


#  ----------------------------------------------------------------------------
#                                   PARSERS
#  ----------------------------------------------------------------------------

class MNISTDefaultParser(TrainableParser):

    def __init__(self, *args, defaults, **kwargs):
        super().__init__(*args, defaults=defaults, **kwargs)
        self.add_argument(
            '--name', type=str, default=defaults['name'],
            help="Name of the model, used in the creation of the model itself "
                 "and the target subdirectories."
        )
        self.add_argument(
            '-aug', '--augment_dataset', action='store_true', default=False,
            help='Create a dataset with elastic transformations.'  # TODO: does this always mean elastic trf?
        )

    def _name_handler(self, args):
        pass

    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)
        self._name_handler(args)
        args.path = Path(args.path) / args.name
        return args


class XcoreTunedParser(MNISTDefaultParser):
    def __init__(self, *args, defaults, **kwargs):
        super().__init__(*args, defaults=defaults, **kwargs)
        self.add_argument(
            '--xcore_tuned', action='store_true', default=False,
            help='Use a variation of the model tuned for xcore.ai.'
        )

    def _name_handler(self, args):
        if args.xcore_tuned:
            args.name = '_'.join([args.name, 'tuned'])
