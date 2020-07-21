# Copyright (c) 2020, XMOS Ltd, All rights reserved

import logging
import tensorflow as tf  # type: ignore

from typing import TYPE_CHECKING, Union, Iterator, List

if TYPE_CHECKING:
    import numpy as np  # type: ignore


def quantize_converter(
    converter: tf.lite.TFLiteConverter,
    representative_data: Union["np.array", tf.Tensor],
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
