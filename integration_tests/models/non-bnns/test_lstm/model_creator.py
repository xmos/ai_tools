import tensorflow as tf
from tensorflow.keras.layers import LSTM, Input
from tensorflow.keras import Model
import numpy as np
import os

def export_lstm_stateful_tflite(units, input_shape):
    main_input = Input(shape=input_shape)
    input_h = Input(shape=(units,))
    input_c = Input(shape=(units,))

    lstm_layer, state_h, state_c = LSTM(units, activation="relu", unroll=True, return_state=True)(main_input, initial_state=[input_h, input_c])
    model = Model(inputs=[main_input, input_h, input_c], outputs=[lstm_layer, state_h, state_c])

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset_gen():
        for _ in range(100):
            yield [np.random.rand(1, *input_shape).astype(np.float32),
                   np.random.rand(1, units).astype(np.float32),
                   np.random.rand(1, units).astype(np.float32)]

    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    model_file = 'test_lstm_0.tflite'
    with open(model_file, 'wb') as f:
        f.write(tflite_model)

    return os.path.getsize(model_file) / 1024

# Example call
example = export_lstm_stateful_tflite(9, (1, 16))
