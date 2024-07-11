
import numpy as np
import tensorflow as tf

BATCH_SIZE = 100
input_shape = (2,)
input_data = tf.keras.Input(shape=input_shape, batch_size=BATCH_SIZE)
print(input_data.shape)

# Apply the Softmax layer
output = tf.keras.layers.Softmax()(input_data)
print(output.shape)

# Create the model
model = tf.keras.Model(inputs=input_data, outputs=output)

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: Define a representative dataset for quantization (not required for this simple model)
def representative_dataset_gen():
    for _ in range(100):
        yield [np.random.uniform(low=-1., high=1., size=(BATCH_SIZE,) + input_shape).astype(np.float32)]

# Optional: Set optimization options (can be commented out if not needed)
converter.representative_dataset = representative_dataset_gen
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
model_name = 'test_softmax_10.tflite'
with open(model_name, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model saved as {model_name}")
