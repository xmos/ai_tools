import numpy as np
import tensorflow as tf
from tensorflow import lite as tfl

i = 0

def generate_slice_model(input_shape, dtype, begin, size):
    begin = [0] + begin
    size = [1] + size
    input_data = tf.keras.Input(shape=input_shape, dtype=dtype, batch_size=1)
    sliced_output = tf.slice(input_data, begin, size)
    model = tf.keras.Model(inputs=input_data, outputs=sliced_output)
    converter = tfl.TFLiteConverter.from_keras_model(model)
    if dtype == tf.int8 or dtype == tf.int16:
        def representative_dataset_gen():
            for _ in range(100):
                yield [np.random.uniform(low=-127, high=127, size=input_shape).astype(dtype.as_numpy_dtype)]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        if dtype == tf.int8:
            converter.target_spec.supported_ops = [tfl.OpsSet.TFLITE_BUILTINS_INT8]
        else:
            converter.target_spec.supported_ops = [tfl.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
        converter.inference_input_type = dtype
        converter.inference_output_type = dtype
    tflite_model = converter.convert()
    global i
    model_name = f'test_slice_{i}.tflite'
    i+=1
    with open(model_name, 'wb') as f:
        f.write(tflite_model)
    print(f'Model saved: {model_name}')


dtypes = [tf.int8, tf.float32]

for dtype in dtypes:
    shape = (10,)
    generate_slice_model(shape, dtype, [0], [5])
    generate_slice_model(shape, dtype, [2], [4])
    generate_slice_model(shape, dtype, [0], [10])
    shape = (8, 16)
    generate_slice_model(shape, dtype, [0, 0], [8, 8])
    generate_slice_model(shape, dtype, [2, 0], [6, 16])
    generate_slice_model(shape, dtype, [4, 1], [4, 15])
    shape = (8, 15, 32)
    generate_slice_model(shape, dtype, [0, 0, 0], [8, 15, 28])
    generate_slice_model(shape, dtype, [2, 0, 0], [6, 15, 32])
    generate_slice_model(shape, dtype, [3, 0, 0], [4, 14, 32])
    shape = (1, 1, 20)
    generate_slice_model(shape, dtype, [0, 0, 0], [1, 1, 15])
    shape = (7, 9, 10, 3)
    generate_slice_model(shape, dtype, [0, 0, 0, 0], [7, 9, 10, 3])
    generate_slice_model(shape, dtype, [1, 5, 0, 0], [6, 3, 10, 3])
    shape = (3, 11, 16, 5)
    generate_slice_model(shape, dtype, [2, 2, 1, 0], [1, 8, 13, 5])
