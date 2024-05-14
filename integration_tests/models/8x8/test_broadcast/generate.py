import numpy as np
import tensorflow as tf
from tensorflow import lite as tfl

i = 0

def generate_broadcast_model(input_shape, output_shape):
    input_data = tf.keras.Input(shape=input_shape, batch_size=1)
    broadcasted_output = tf.broadcast_to(input_data, (1,)+output_shape)
    model = tf.keras.Model(inputs=input_data, outputs=broadcasted_output)
    converter = tfl.TFLiteConverter.from_keras_model(model)
    def representative_dataset_gen():
        for _ in range(100):
            yield [np.random.uniform(low=-1, high=1, size=input_shape).astype(np.float32)]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tfl.OpsSet.TFLITE_BUILTINS_INT8]
    tflite_model = converter.convert()
    global i
    model_name = f'test_broadcast_{i}.tflite'
    i+=1
    with open(model_name, 'wb') as f:
        f.write(tflite_model)
    print(f'Model saved: {model_name}')


generate_broadcast_model((5, 1, 7),  (5, 8, 7))
generate_broadcast_model((5, 1, 16), (5, 8, 16))
generate_broadcast_model((1, 1, 32), (5, 8, 32))
generate_broadcast_model((5, 1, 4),  (5, 8, 4))
generate_broadcast_model((5, 1),     (5, 3))
