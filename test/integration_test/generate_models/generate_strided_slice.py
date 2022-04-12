import tensorflow as tf
import numpy as np
import os
from math import ceil
import yaml

dir = os.path.dirname(__file__)

with open(os.path.join(dir,'test_strided_slice.yml'), 'r') as file:
	configs = yaml.safe_load(file)

height = configs['default'][0]['height']
width = configs['default'][0]['width']
channels = configs['default'][0]['channels']

inputs = tf.keras.Input(shape=(height,width,channels),)
x_0 = tf.strided_slice(inputs,begin=[0, 0, 0], end= [height, width//2, channels],strides= [1, 1, 1])
x_1 = tf.strided_slice(inputs,begin=[0, ceil(width/2), 0], end= [height, width, channels],strides= [1, 1, 1])
x = tf.keras.layers.Concatenate(axis=1)([x_0,x_1])
outputs = tf.keras.layers.Flatten()(x)
model = tf.keras.Model(inputs=inputs,outputs=[outputs])

saved_model_dir = os.path.join(dir,"saved_model")
model.save(saved_model_dir)

def representative_dataset():
    for _ in range(100):
      data = np.random.rand(1,height,width,channels)
      yield [data.astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_quant_model = converter.convert()

with open(os.path.join(dir,"strided_slice.tflite"),mode='wb') as file:
  file.write(tflite_quant_model)
