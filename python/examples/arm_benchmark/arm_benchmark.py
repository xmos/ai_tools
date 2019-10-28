#!/usr/bin/env python
#
# Copyright (c) 2019, XMOS Ltd, All rights reserved

# always load examples_common first to avoid debug info dump from tf initialization
import examples.examples_common as common

import argparse
import datetime
import logging
import pathlib
import tflite_utils

import tensorflow as tf
import numpy as np
import tflite2xcore_graph_conv as graph_conv

from tensorflow import keras
from copy import deepcopy


DIRNAME = pathlib.Path(__file__).parent
MODELS_DIR, DATA_DIR = common.make_aux_dirs(DIRNAME)
SEED = 1234


def get_normalized_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    scale = tf.constant(255, dtype=tf.dtypes.float32)
    x_train, x_test = train_images/scale - .5, test_images/scale - .5
    y_train, y_test = train_labels, test_labels

    return {'x_train': np.float32(x_train), 'y_train': np.float32(y_train),
            'x_test': np.float32(x_test), 'y_test': np.float32(y_test)}


def create_data_generator(x_train):
    tflite_utils.set_all_seeds(seed=SEED)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)
    datagen.fit(x_train)
    return datagen


def build_model():
    return keras.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=5,
                               padding='same', input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        tf.keras.layers.Conv2D(filters=64, kernel_size=5, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])


def main(*, train_new_model=False, evaluate_models=False):

    data_path = DATA_DIR / 'training_dataset.npz'
    keras_model_path = MODELS_DIR / "model.h5"

    # load data
    data = get_normalized_data()

    if train_new_model:
        keras.backend.clear_session()
        tflite_utils.set_all_seeds(seed=SEED)

        # create model
        model = build_model()
        training_params = {'optimizer': 'adam',
                           'loss': 'sparse_categorical_crossentropy',
                           'metrics': ['accuracy']}
        model.compile(**training_params)

        # create generator
        datagen = create_data_generator(data['x_train'])

        # run the training
        batch_size = 32
        epochs = 30
        USE_TENSORBOARD = False

        # run the training
        if USE_TENSORBOARD:
            # TODO: this is untested, may not work
            log_dir = (DIRNAME / "logs") / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=0)

            tflite_utils.set_all_seeds(seed=SEED)
            model.fit_generator(datagen.flow(data['x_train'], data['y_train'], batch_size=batch_size),
                                epochs=1,
                                validation_data=(data['x_test'], data['y_test']),
                                callbacks=[tensorboard_callback])
            model.fit_generator(datagen.flow(data['x_train'], data['y_train'], batch_size=batch_size),
                                epochs=epochs-1,
                                validation_data=(data['x_test'], data['y_test']),
                                callbacks=[tensorboard_callback])
        else:
            tflite_utils.set_all_seeds(seed=SEED)
            model.fit_generator(datagen.flow(data['x_train'], data['y_train'], batch_size=batch_size),
                                epochs=epochs,
                                validation_data=(data['x_test'], data['y_test']))

        # save model and data
        np.savez(data_path, **data)
        model.save(keras_model_path)

    else:
        try:
            logging.info(f"Loading keras from {keras_model_path}")
            model = keras.models.load_model(keras_model_path)
        except FileNotFoundError as e:
            logging.error(f"{e} (Hint: use the --train_model flag)")
            return

    # choose test data examples
    sorted_inds = np.argsort(data['y_test'], axis=0, kind='mergesort')
    subset_inds = np.searchsorted(data['y_test'][sorted_inds].flatten(), np.arange(10))
    subset_inds = sorted_inds[subset_inds]
    x_test_float = data['x_test'][subset_inds.flatten()]

    # convert to TFLite float, save model and visualization, save test data
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    common.save_from_tflite_converter(converter, MODELS_DIR, "model_float")
    common.save_test_data_for_converter(
        converter, x_test_float, data_dir=DATA_DIR, base_file_name="model_float")

    # optionally evaluate float model
    if evaluate_models:
        logging.info(f"Evaluating model_float on test set...")
        interpreter = tf.lite.Interpreter(model_content=converter.convert())
        y = common.apply_interpreter_to_examples(interpreter, data['x_test'], show_progress_step=1000)
        y = np.argmax(np.vstack(y), axis=1).flatten()
        correct = np.sum(data['y_test'].flatten() == y)
        acc = correct / y.size
        print(f"model_float accuracy: {acc:.2%}")


    # convert to TFLite quantized, save model and visualization, save test data
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    common.quantize_converter(converter, data['x_train'])
    model_quant_file = common.save_from_tflite_converter(converter, MODELS_DIR, "model_quant")
    common.save_test_data_for_converter(
        converter, x_test_float, data_dir=DATA_DIR, base_file_name="model_quant")

    # optionally evaluate quant model
    if evaluate_models:
        # TODO: parallelize this
        logging.info(f"Evaluating model_quant on test set...")
        interpreter = tf.lite.Interpreter(model_content=converter.convert())
        y = common.apply_interpreter_to_examples(interpreter, data['x_test'], show_progress_step=50)
        y = np.argmax(np.vstack(y), axis=1).flatten()
        correct = np.sum(data['y_test'].flatten() == y)
        acc = correct / y.size
        print(f"model_quant accuracy: {acc:.2%}")

    # load quantized model in json, serving as basis for conversions
    # strip quantized model of float interface and softmax
    model_quant = tflite_utils.load_tflite_as_json(model_quant_file)
    model_stripped = common.strip_model_quant(model_quant)
    model_stripped['description'] = "TOCO Converted and stripped."
    common.save_from_json(model_stripped, MODELS_DIR, 'model_stripped')
    common.save_test_data_for_stripped_model(
        model_stripped, x_test_float, data_dir=DATA_DIR)

    # save xcore converted model
    model_xcore = deepcopy(model_quant)
    graph_conv.convert_model(model_xcore, remove_softmax=True)
    common.save_from_json(model_xcore, MODELS_DIR, 'model_xcore')
    common.save_test_data_for_xcore_model(
        model_xcore, x_test_float, data_dir=DATA_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', action='store_true', default=False,
                        help='Use GPU for training. Might result in non-reproducible results')
    parser.add_argument('--train_model', action='store_true', default=False,
                        help='Train new model instead of loading pretrained tf.keras model.')
    parser.add_argument('--evaluate_models', action='store_true', default=False,
                        help='Evaluate models on test set.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Verbose mode.')
    args = parser.parse_args()

    verbose = args.verbose

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.getLogger('tensorflow').setLevel(logging.ERROR)

    logging.info(f"Eager execution enabled: {tf.executing_eagerly()}")

    tflite_utils.set_gpu_usage(args.use_gpu, verbose)

    main(train_new_model=args.train_model, evaluate_models=args.evaluate_models)
