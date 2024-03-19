import numpy as np
import tensorflow as tf
from tensorflow import lite as tfl

i = 0

def generate_concatenate_model(input_shapes, axis):
    input_data = [tf.keras.Input(shape=input_shape, dtype=np.float32, batch_size=1) for input_shape in input_shapes]
    concatenated_output = tf.keras.layers.Concatenate(axis=axis)(input_data)
    model = tf.keras.Model(inputs=input_data, outputs=concatenated_output)
    converter = tfl.TFLiteConverter.from_keras_model(model)
    def representative_dataset_gen():
        for _ in range(100):
            yield [np.random.uniform(low=-127, high=1270, size=shp).astype(np.float32) for shp in input_shapes]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tfl.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    converter._experimental_full_integer_quantization_bias_type=tf.int32

    tflite_model = converter.convert()
    global i
    model_name = f'test_concatenate_{i}.tflite'
    i+=1
    with open(model_name, 'wb') as f:
        f.write(tflite_model)
    print(f'Model saved: {model_name}')


generate_concatenate_model([(64), (64)], 0)
generate_concatenate_model([(2, 3), (2, 3)], 1)
generate_concatenate_model([(2, 3, 5), (2, 3, 5)], 0)
generate_concatenate_model([(2, 6, 5, 2), (2, 6, 5, 2)], 1)

