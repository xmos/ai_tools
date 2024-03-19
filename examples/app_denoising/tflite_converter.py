import tensorflow as tf
from model import get_trunet
from xmos_ai_tools import xformer
from dataset import data_gen
from tqdm import tqdm
import numpy as np


def get_rep_dataset(model):
    d = data_gen("data/datasets_fullband/",
                 "data/MS-SNSD/", "data/rirs_noises/")
    train_sample, _ = d.__next__()
    outputs = np.zeros(train_sample.shape)
    states = np.zeros([len(outputs), 64], dtype=np.float32)
    for i in tqdm(range(len(outputs))):
        out, state = model([train_sample[i:i+1][None], states[i:i+1]])
        outputs[i:i+1] = out
        if i != len(outputs) - 1:
            states[i+1:i+2] = state
    for t, s in zip(train_sample[:, None, None], states[:, None]):
        yield [t, s]


def save_tflite(model, ws_path, quant="16x8"):
    model.load_weights(ws_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.representative_dataset = lambda: get_rep_dataset(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if quant == "16x8":
        print("Using experimental 16x8 quantization...")
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
        converter.inference_input_type = tf.int16
        converter.inference_output_type = tf.int16
    elif quant == "8x8":
        print("Using 8x8 quantization...")
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        print("Using float32...")
    return converter.convert()


def save_xformed(model_in, model_out):
    hyper_params = {"xcore-thread-count": 5}
    xformer.convert(model_in, model_out, hyper_params)


if __name__ == "__main__":
    USE_XINTERPRETER = False
    QUANT_TYPE = "float32"
    MODEL_PATH = "models/model_64f_114k.h5"
    OUTPUT_XC_MODEL = f"models/model_xc_{QUANT_TYPE}.tflite"
    OUTPUT_TFLITE_MODEL = f"models/model_{QUANT_TYPE}.tflite"
    model = get_trunet(64, 1, True)
    tflite_model = save_tflite(model, MODEL_PATH, quant=QUANT_TYPE)
    with open(OUTPUT_TFLITE_MODEL, "wb") as f:
        f.write(tflite_model)
    if USE_XINTERPRETER:
        save_xformed(OUTPUT_TFLITE_MODEL, OUTPUT_XC_MODEL)
