# Copyright (c) 2018-2020, XMOS Ltd, All rights reserved

import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np

from tflite2xcore.utils import tf  # ensure that tf is imported lazily
from tflite2xcore import xlogging as logging


def quantize(arr, scale, zero_point, dtype=np.int8):
    t = np.round(arr / scale + zero_point)
    return dtype(np.round(np.clip(t, np.iinfo(dtype).min, np.iinfo(dtype).max)))


def dequantize(arr, scale, zero_point):
    return np.float32((arr.astype(np.int32) - np.int32(zero_point)) * scale)


def quantize_converter(converter, representative_data):
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
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
    interpreter.allocate_tensors()
    if interpreter_input_ind is None:
        interpreter_input_ind = interpreter.get_input_details()[0]["index"]
    if interpreter_output_ind is None:
        interpreter_output_ind = interpreter.get_output_details()[0]["index"]

    outputs = []
    for j, x in enumerate(examples):
        if show_progress_step and (j+1) % show_progress_step == 0:
            if show_pid:
                logging.getLogger().info(f"(PID {os.getpid()}) Evaluated examples {j+1:6d}/{examples.shape[0]}")
            else:
                logging.getLogger().info(f"Evaluated examples {j+1:6d}/{examples.shape[0]}")
        interpreter.set_tensor(interpreter_input_ind, np.expand_dims(x, 0))
        interpreter.invoke()
        y = interpreter.get_tensor(interpreter_output_ind)
        outputs.append(y)

    return np.vstack(outputs) if isinstance(examples, np.ndarray) else outputs


def plot_history(history, title='metrics', zoom=1, save=False, path=None):
    path = path or pathlib.Path('./history.png')
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
