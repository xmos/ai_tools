import numpy as np
import tensorflow as tf

input_shape = (8, 1, 9)
input_data = tf.keras.Input(shape=input_shape, dtype=tf.int8, batch_size=1)
sliced_output = input_data[:, ::-2, 0, :-5]
model = tf.keras.Model(inputs=input_data, outputs=sliced_output)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
def representative_dataset_gen():
    for _ in range(100):
        yield [np.random.uniform(low=-127, high=127, size=input_shape).astype(np.int8)]
converter.representative_dataset = representative_dataset_gen
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
model_name = f'strided_slice_5.tflite'
with open(model_name, 'wb') as f:
    f.write(tflite_model)
