import tensorflow as tf


def save_quantized_mobilenet(model: tf.keras.Model, model_path: str, size: tuple):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Use a representative dataset generator for accurate activation quantization
    rep_ds = tf.data.Dataset.list_files("image_samples/*.jpg")

    def representative_dataset_gen():
        for image_path in rep_ds:
            img = tf.io.read_file(image_path)
            img = tf.io.decode_image(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            resized_img = tf.image.resize(img, size)
            resized_img = resized_img[tf.newaxis, :]
            yield [resized_img]

    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(model_path, "wb") as f:
        f.write(tflite_model)
    print("Model quantized and saved successfully.")
