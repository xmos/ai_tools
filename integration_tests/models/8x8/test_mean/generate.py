import numpy as np
import tensorflow as tf
from tensorflow import lite as tfl

i = 0


def generate_mean_model(input_shape, axes):
    input_data = tf.keras.Input(shape=input_shape, dtype=tf.int8, batch_size=1)
    mean_output = tf.keras.backend.mean(input_data, axis=axes)
    model = tf.keras.Model(inputs=input_data, outputs=mean_output)
    converter = tfl.TFLiteConverter.from_keras_model(model)

    def representative_dataset_gen():
        for _ in range(100):
            yield [
                np.random.uniform(low=-127, high=127, size=input_shape).astype(np.int8)
            ]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tfl.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    global i
    model_name = f"test_mean_{i}.tflite"
    i += 1
    with open(model_name, "wb") as f:
        f.write(tflite_model)
    print(f"Model saved: {model_name}")


input_shapes_and_axes = [
    ((10,), [0]),
    ((8, 16), [0]),
    ((8, 16), [1]),
    ((8, 16), [0, 1]),
    ((8, 15, 32), [0]),
    ((8, 15, 32), [1]),
    ((8, 15, 32), [2]),
    ((8, 15, 32), [0, 2]),
]

for shape, axes in input_shapes_and_axes:
    generate_mean_model(shape, axes)
