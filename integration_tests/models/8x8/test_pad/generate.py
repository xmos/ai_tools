import numpy as np
import tensorflow as tf

input_shape = (1, 8, 13, 5)
input_data = tf.keras.Input(shape=input_shape, batch_size=1)
print(input_data.shape)
l = tf.keras.layers.ZeroPadding3D(padding=((2, 0), (2, 1), (1, 2)), data_format='channels_last')
# l = tf.keras.layers.Conv2D(8, 5, strides=[2,2], padding='same')
output = l(input_data)
# l.set_weights([np.zeros((5, 5, 9, 8)), np.zeros(8)])
print(output.shape)
model = tf.keras.Model(inputs=input_data, outputs=output)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
def representative_dataset_gen():
    for _ in range(100):
        yield [np.random.uniform(low=-1., high=1., size=(1,) + input_shape).astype(np.float32)]
converter.representative_dataset = representative_dataset_gen
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
model_name = f'test_pad_30.tflite'
with open(model_name, 'wb') as f:
    f.write(tflite_model)
