# Copyright (c) 2018-2020, XMOS Ltd, All rights reserved

from tflite2xcore.utils import (
    set_all_seeds, set_gpu_usage, set_verbosity, LoggingContext
)

import os
import random
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm


def quantize(arr, scale, zero_point, dtype=np.int8):
    t = np.round(arr / scale + zero_point)
    return dtype(np.round(np.clip(t, np.iinfo(dtype).min, np.iinfo(dtype).max)))


def quantize_converter(converter, representative_data):
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    x_train_ds = tf.data.Dataset.from_tensor_slices(representative_data).batch(1)

    def representative_data_gen():
        for input_value in x_train_ds.take(representative_data.shape[0]):
            yield [input_value]
    converter.representative_dataset = representative_data_gen


def apply_interpreter_to_examples(interpreter, examples, *,
                                  interpreter_input_ind=None,
                                  interpreter_output_ind=None,
                                  show_progress_step=None,
                                  show_pid=False):
    if interpreter_input_ind is None:
        interpreter_input_ind = interpreter.get_input_details()[0]["index"]
    if interpreter_output_ind is None:
        interpreter_output_ind = interpreter.get_output_details()[0]["index"]
    interpreter.allocate_tensors()

    outputs = []
    for j, x in enumerate(examples):
        if show_progress_step and (j+1) % show_progress_step == 0:
            if show_pid:
                logging.info(f"(PID {os.getpid()}) Evaluated examples {j+1:6d}/{examples.shape[0]}")
            else:
                logging.info(f"Evaluated examples {j+1:6d}/{examples.shape[0]}")
        interpreter.set_tensor(interpreter_input_ind, tf.expand_dims(x, 0))
        interpreter.invoke()
        y = interpreter.get_tensor(interpreter_output_ind)
        outputs.append(y)

    return np.vstack(outputs) if isinstance(examples, np.ndarray) else outputs


def shuffle(arr1, arr2):
    assert len(arr1) == len(arr2), 'Arrays must be same length'
    ind_list = [i for i in range(len(arr1))]
    random.shuffle(ind_list)
    train_new = arr1[ind_list, :, :, :]
    target_new = arr2[ind_list, ]
    assert train_new.shape == arr1.shape
    assert target_new.shape == arr2.shape
    return train_new, target_new


def unfold_gen(size, generator):
    '''
    To unfold a numpy generator, need to be fed with size.
    \t- size: expected size of the unfolded object
    \t- generator: generator object to unfold
    '''
    arr = np.empty(size)
    for i, el in enumerate(generator):
        arr[i] = el
    return arr


def _flatten(ds):
    '''
    Flatten function for a numpy array. It must have 3 dimensions,
    and the output will have 2.
    '''
    return ds.reshape(ds.shape[0], ds.shape[1]*ds.shape[2])


def save_data_to_file(path, x, y, xt=0, yt=0):
    '''
    Will save a numpy dictionary in the path provided.
    '''
    data = {}
    data['x_train'] = x
    data['y_train'] = y
    if len(xt) != 0 and len(yt) != 0:
        data['x_test'] = xt
        data['y_test'] = yt
    np.savez(path, **data)


# TODO: move this to MNIST specific utils file
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
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
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
        y_train = to_categorical(y_train, nb_classes)
        y_test = to_categorical(y_test, nb_classes)
        y_train = y_train.reshape(y_train.shape[0], 10)
        y_test = y_test.reshape(y_test.shape[0], 10)

    if val_split:
        index = int(0.8 * len(x_train))
        x_train, x_val = x_train[:index], x_train[index:]
        y_train, y_val = y_train[:index], y_train[index:]

    if flatten:
        x_train = _flatten(x_train)
        x_test = _flatten(x_test)
        if val_split:
            x_val = _flatten(x_val)
    if not categorical:
        train_labels_count = np.unique(y_train, return_counts=True)
        logging.debug(f"labels counts: {train_labels_count[1]}")
    if val_split:
        return x_train, x_test, x_val, y_train, y_test, y_val
    return x_train, x_test, y_train, y_test


# TODO: this takes a while, add progress bar
# TODO: change name to something more meaningful
def ecc(nsizex=29, nsizey=29, ch=1):
    '''
    Crop the dataset images using resize from skimage,
    consider instead use keras layer Cropping2D.
    '''
    x_train, x_test, x_val, y_train, y_test, y_val = get_mnist(
        padding=0, categorical=False)
    from skimage.transform import resize
    with tqdm(total=30) as pbar:
        o_train = resize(x_train, (x_train.shape[0], nsizex, nsizey, ch))
        pbar.update(10)
        o_test = resize(x_test, (x_test.shape[0], nsizex, nsizey, ch))
        pbar.update(10)
        o_val = resize(x_val, (x_val.shape[0], nsizex, nsizey, ch))
        pbar.update(10)
    return o_train, o_test, o_val, y_train, y_test, y_val


# TODO: move this to MNIST specific utils file
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


# Viz
def sanity_check(ds, labels):
    '''
    Show a random image to perform a sanity check of the data.
    \t- ds: dataset (numpy array)
    \t- labels: dataset labes (numpy array)
    '''
    idx = random.randint(0, len(ds))
    img = ds[idx].squeeze()
    plt.style.use('dark_background')
    plt.figure(figsize=(1, 1))
    plt.title('Index: ' + str(labels[idx]) + ' - sanity check')
    plt.imshow(img)


def random_pick(ds, labels, categorical=False, dim=32, ch=1, zoom=1):
    '''
    Show and return a random image from a given dataset.
    \t- ds: dataset (numpy array)
    \t- labels: dataset labels (numpy array)
    \t- categorical: if labels are in categorical format
    \t- dim: dimension of the side of the image
    \t- ch: number of channels
    \t- zoom: zoom for the plotting
    '''
    idx = random.randint(0, len(ds))
    exp = labels[idx]
    if categorical:
        exp = np.argmax(exp)
    plt.style.use('dark_background')
    plt.figure(figsize=(zoom, zoom))
    plt.title('Index: ' + str(exp) + ' - random pick')
    plt.imshow(ds[idx].reshape(dim, dim, ch).squeeze())
    return ds[idx]


def random_stack(ds, labels, depth, categorical=False, dim=32, ch=1):
    '''
    Return a random stack of a given dataset.
    \t- ds: dataset (numpy array)
    \t- labels: dataset labels (numpy array)
    \t- depth: number of data instances in the stack
    \t- categorical: if labels are in categorical format
    \t- dim: dimension of the side of the image
    \t- ch: number of channels
    '''
    stack = np.row_stack([
        random_pick(ds, labels, categorical, dim, ch) for _ in range(depth)
    ])
    logging.debug(f"random stack shape: {stack.shape}")
    return stack


def plot(img, title='', zoom=3, dim=32, ch=1):
    '''
    Plot easily an image using matplotlib.
    \t- img: image to plot
    \t- title: title of the plot
    \t- zoom: zoom for the plot
    \t- dim: dimension of the side of the image
    \t- ch: number of channels of the image
    '''
    plt.style.use('dark_background')
    plt.figure(figsize=(zoom, zoom))
    plt.title(title)
    plt.imshow(img.reshape(dim, dim, ch).squeeze())


def multi_plot(imgs, rows, cols, title='', zoom=2):
    '''
    Plot several images easily using matplotlib
    \t- imgs: stack of images to be plotted
    \t- rows: number of rows of the output grid
    \t- cols: number of cols of the output grid
    \t- title: title of the grid
    \t- zoom: zoom of the images in the grid
    '''
    assert rows*cols >= len(imgs)
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(8*zoom, 8*zoom))
    for i in range(1, rows*cols + 1):
        img = imgs[i-1]
        plt.title(title)
        fig.add_subplot(rows, cols, i)
        plt.imshow(img.squeeze())
    plt.show()


def plot_history(history, title='metrics', zoom=1, save=False, path=Path('./history.png')):
    # list all data in history
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16*zoom, 8*zoom))
    plt.title(title)
    plt.axis('off')

    # summarize history for accuracy
    fig.add_subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    fig.add_subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # Save the png
    fig.savefig(path)


# Augmentation
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
