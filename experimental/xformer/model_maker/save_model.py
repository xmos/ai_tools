import tensorflow as tf
import pathlib


# image_shape = (224,224,3)
# def representative_dataset_gen():
#     for i in range(100):
#         # creating fake images
#         image = tf.random.normal([1] + list(image_shape))
#         yield [image]

rep_ds = tf.data.Dataset.list_files("train_samples/*.jpg")
HEIGHT, WIDTH = 224, 224

def representative_dataset_gen():
   for image_path in rep_ds:
       img = tf.io.read_file(image_path)
       img = tf.io.decode_image(img, channels=3)
       img = tf.image.convert_image_dtype(img, tf.float32)
       resized_img = tf.image.resize(img, (HEIGHT, WIDTH))
       resized_img = resized_img[tf.newaxis, :]
       yield [resized_img]

model = tf.keras.applications.mobilenet.MobileNet(alpha=0.25)
converter = tf.lite.TFLiteConverter.from_keras_model(model)


converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
#converter.representative_dataset = tf.lite.RepresentativeDataset(representative_dataset_gen)
converter.representative_dataset = representative_dataset_gen
model_tflite = converter.convert()

tflite_model = converter.convert()

tflite_model_file = pathlib.Path("pretrainedmodel.tflite")
tflite_model_file.write_bytes(tflite_model)
