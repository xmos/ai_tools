import tensorflow as tf
import numpy as np
from tqdm import tqdm
from scipy.signal import istft
from scipy.io import wavfile
from dataset import F_BANK, process_wave, read_tfrecords
from xmos_ai_tools.xinterpreters import TFLMHostInterpreter
from train import weighted_mse


def reconstruct_signal(magnitudes, phases, fs=16000):
    complex_spectrogram = magnitudes * np.exp(1j * phases)
    _, reconstructed_signal = istft(
        complex_spectrogram.T,
        fs=16000,
        nperseg=512,
        noverlap=512 - 128,
        nfft=512
    )
    return reconstructed_signal


def get_xinterpreter(model_path):
    with open(model_path, "rb") as fd:
        model = fd.read()
    ie = TFLMHostInterpreter()
    ie.set_model(model_content=model, secondary_memory=False)
    return ie


def get_tflite_interpreter(model_path):
    ie = tf.lite.Interpreter(model_path=model_path)
    ie.allocate_tensors()
    return ie


def get_preds(ie, x):
    in_dets = ie.get_input_details()
    out_dets = ie.get_output_details()
    in_scale, in_zp = in_dets[0]['quantization']
    out_scale, out_zp = out_dets[1]['quantization']
    if in_scale == 0. and out_scale == 0.:
        in_scale = out_scale = 1.
    x = (x / in_scale + in_zp).astype(in_dets[0]["dtype"])
    outputs = np.zeros(x.shape, dtype=out_dets[1]["dtype"])
    state = np.zeros([1, 64], dtype=out_dets[0]["dtype"])
    for i in range(len(outputs)):
        ie.set_tensor(in_dets[1]["index"], state)
        ie.set_tensor(in_dets[0]["index"], x[i:i+1][None])
        ie.invoke()
        state = ie.get_tensor(out_dets[0]['index'])
        outputs[i:i+1] = ie.get_tensor(out_dets[1]['index'])
    outputs = ((outputs.astype(np.float32) - out_zp) * out_scale)
    return outputs[..., 0]


def write_wav(signal, path):
    signal = (signal / np.max(np.abs(signal)) * (2**14)).astype(np.int16)
    wavfile.write(path, 16000, signal)


def evaluate_model(ie):
    _, test = read_tfrecords("data/records_64/", bs=1)
    losses = []
    for x, y in tqdm(test):
        preds = get_preds(ie, x.numpy()[0])[None, ..., None]
        losses.append(weighted_mse(y, preds*x))
    return np.mean(losses)


if __name__ == "__main__":
    QUANT_TYPE = "float32"
    MODEL_NAME = f"models/model_{QUANT_TYPE}.tflite"
    USE_XINTERPRETER = False
    if USE_XINTERPRETER:
        ie = get_xinterpreter(MODEL_NAME)
    else:
        ie = get_tflite_interpreter(MODEL_NAME)
    print(evaluate_model(ie))
    for num in range(1, 7):
        input_path = f"samples/input_{num}.wav"
        output_path = f"samples/output_{num}_{QUANT_TYPE}_relu.wav"
        _, wav = wavfile.read(input_path)
        x, orig = process_wave(wav, True)
        outs = get_preds(ie, x)
        mask = ((outs @ F_BANK)**(1/.3)).clip(0.1, 1)
        mags = (np.abs(orig) * mask)
        signal = reconstruct_signal(mags, np.angle(orig))
        write_wav(signal, output_path)
