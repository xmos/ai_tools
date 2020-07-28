# Copyright (c) 2020, XMOS Ltd, All rights reserved

import os
import logging
import tensorflow as tf  # type: ignore
import numpy as np  # type: ignore

from typing import TYPE_CHECKING, Union, Iterator, List, Optional


def quantize_converter(
    converter: tf.lite.TFLiteConverter,
    representative_data: Union[tf.Tensor, "np.ndarray"],
    *,
    show_progress_step: int = 0,
) -> None:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    x_train_ds = tf.data.Dataset.from_tensor_slices(representative_data).batch(1)

    def representative_data_gen() -> Iterator[List[tf.Tensor]]:
        for j, input_value in enumerate(x_train_ds.take(representative_data.shape[0])):
            if show_progress_step and (j + 1) % show_progress_step == 0:
                logging.getLogger().info(
                    f"Converter quantization processed examples {j+1:6d}/{representative_data.shape[0]}"
                )
            yield [input_value]

    converter.representative_dataset = representative_data_gen


def apply_interpreter_to_examples(
    interpreter: tf.lite.Interpreter,
    examples: Union[tf.Tensor, "np.ndarray"],
    *,
    interpreter_input_ind: Optional[int] = None,
    interpreter_output_ind: Optional[int] = None,
    show_progress_step: int = 0,
    show_pid: bool = False,
) -> "np.ndarray":
    interpreter.allocate_tensors()
    if interpreter_input_ind is None:
        interpreter_input_ind = interpreter.get_input_details()[0]["index"]
    if interpreter_output_ind is None:
        interpreter_output_ind = interpreter.get_output_details()[0]["index"]

    outputs = []
    for j, x in enumerate(examples):
        if show_progress_step and (j + 1) % show_progress_step == 0:
            if show_pid:
                logging.getLogger().info(
                    f"(PID {os.getpid()}) Evaluated examples {j+1:6d}/{examples.shape[0]}"
                )
            else:
                logging.getLogger().info(
                    f"Evaluated examples {j+1:6d}/{examples.shape[0]}"
                )
        interpreter.set_tensor(interpreter_input_ind, np.expand_dims(x, 0))
        interpreter.invoke()
        y = interpreter.get_tensor(interpreter_output_ind)
        outputs.append(y)

    return np.vstack(outputs) if isinstance(examples, np.ndarray) else outputs
