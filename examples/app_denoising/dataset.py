import os
import tensorflow as tf
import random
from tqdm import tqdm
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, resample, fftconvolve
import noisereduce as nr
from glob import glob

SAMPLES = None
WINDOW_SIZE = 512
HOP_SIZE = 128
FFT_SIZE = 512
FL = 512 * 8 * 40
FADE_IN_LENGTH = 512 * 50
SR = 16000
NUM_BINS = 64
POWER_FACTOR = .3
tf.keras.utils.set_random_seed(42)


def unique_log_bins(low, high, nbins):
    if low < 1:
        bins = np.geomspace(1, high, nbins-1, dtype=int)
        bins = np.concatenate(([0], bins))
    else:
        bins = np.geomspace(low, high, nbins, dtype=int)
    while len(np.unique(bins)) != nbins:
        unique_vals, counts = np.unique(bins, return_counts=True)
        duplicates = np.argwhere(counts > 1)
        arg_first_unique = duplicates[-1][0] + 1
        first_unique = unique_vals[arg_first_unique]
        total_duplicates = np.sum(unique_vals < first_unique)
        next_bins = np.geomspace(
            first_unique, high, nbins - total_duplicates, dtype=int)
        bins = np.concatenate((unique_vals[:arg_first_unique], next_bins))
    return bins


def log_filterbank(Fs, nfft, n_filters=24, f_low=0, f_high=None, window_function=np.hanning):
    f_high = f_high or Fs/2.0
    assert (f_high <= Fs/2.0), "Log filterbank higher frequency cannot exceed Fs/2!"
    bin_low = np.floor(f_low*(nfft)/Fs)
    bin_high = np.floor(f_high*(nfft)/Fs)
    Hz_bins = unique_log_bins(bin_low, bin_high, n_filters)
    fbank = np.zeros([n_filters, nfft // 2 + 1])
    for n in range(n_filters-1):
        dist = int(Hz_bins[n+1] - Hz_bins[n])
        wind = window_function(2*dist + 1)
        fbank[n, Hz_bins[n]:Hz_bins[n+1]] = wind[dist:-1]
        fbank[n+1, Hz_bins[n]:Hz_bins[n+1]] = wind[:dist]
    fbank[0, :Hz_bins[0]] = 1.0
    fbank[-1, Hz_bins[-1]:] = 1.0
    return fbank


F_BANK = log_filterbank(SR, WINDOW_SIZE, NUM_BINS)
_, NOISE_AUDIO = wavfile.read("data/noise.wav")


def infinite(gen, *args, **kwargs):
    while True:
        yield from gen(*args, **kwargs)


def nsf(signal, noise, snr):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    return np.sqrt((signal_power / noise_power) * 10 ** (-snr / 10.0))


def apply_reverb(signal, rir):
    ratio = np.random.uniform(0, 1)
    r_signal = fftconvolve(signal, rir, mode="full")[:len(signal)]
    return ratio * r_signal + (1.-ratio) * signal


def process_wave(signal, return_orig=False):
    _, _, s = stft(signal, fs=SR, nperseg=WINDOW_SIZE,
                   noverlap=WINDOW_SIZE - HOP_SIZE, nfft=FFT_SIZE)
    mag = np.abs(s.T) @ F_BANK.T
    mag = mag[..., None].astype(np.float32)**POWER_FACTOR
    if return_orig:
        return mag, s.T
    return mag


def pad(sig):
    tot_pad = FL - len(sig)
    left_pad = np.random.randint(0, tot_pad + 1)
    right_pad = tot_pad - left_pad
    return np.concatenate([np.zeros(left_pad), sig, np.zeros(right_pad)])


def get_input(signal, noise, rirs, return_phase=False):
    snr = np.random.uniform(0, 30)
    noise_factor = nsf(signal, noise, snr)
    if len(signal) > FADE_IN_LENGTH:
        signal[:FADE_IN_LENGTH] *= np.arange(0, 1, 1/FADE_IN_LENGTH)
    signal, noise = pad(signal), pad(noise)
    r_signal = apply_reverb(signal, rirs[..., 0])
    r_noise = apply_reverb(noise, rirs[..., 1])
    noisy_signal = r_signal + r_noise * noise_factor
    if return_phase:
        ins, orig = process_wave(noisy_signal, True)
        outs, perf = process_wave(signal, True)
        return ins, outs, orig, perf
    else:
        ins, outs = process_wave(noisy_signal), process_wave(signal)
        return ins, outs


def signal_gen(folder, chop=True, is_clean=False):
    paths = glob(f"{folder}/**/*.wav", recursive=True)
    random.shuffle(paths)
    for path in paths:
        fs, s = wavfile.read(path)
        if is_clean:
            s = nr.reduce_noise(s, sr=fs, y_noise=NOISE_AUDIO,
                                stationary=True, n_fft=512,
                                time_mask_smooth_ms=32,
                                freq_mask_smooth_hz=188,
                                n_std_thresh_stationary=.8)
        s = resample(s, int(len(s) / fs * SR))
        if chop:
            yield from (s[i:i+FL] for i in range(0, len(s), FL))
        elif s.shape[0] <= FL and len(s.shape) == 2:
            yield from (s[:, i-2:i] for i in range(2, s.shape[1], 2))


def chop_silence(wav):
    aw = np.abs(wav[..., 0])
    maw = np.max(aw)
    index = np.where(aw / maw > 0.7)[0][0]
    return wav[index:] / maw


def data_gen(sig_fol, noise_fol, rir_fol, phase=False):
    voices = signal_gen(sig_fol, is_clean=True)
    noises = infinite(signal_gen, noise_fol)
    rirs = map(chop_silence, infinite(signal_gen, rir_fol, False))
    combined = zip(voices, noises, rirs)
    yield from (get_input(s, n, r, phase) for s, n, r in combined)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()]))


def serialize_example(signal, noise):
    feature = {
        'signal': _bytes_feature(signal),
        'noise': _bytes_feature(noise),
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def write_tfrecords(folder_path, data_generator, samples_per_file=256):
    file_count = samples_written = 0
    tfrecord_writer = None
    for signal, noise in tqdm(data_generator):
        if not samples_written:
            file_name = f"{folder_path}/data_{file_count}.tfrecord"
            tfrecord_writer = tf.io.TFRecordWriter(file_name)
        tf_example = serialize_example(signal, noise)
        tfrecord_writer.write(tf_example)
        samples_written += 1
        if samples_written == samples_per_file:
            tfrecord_writer.close()
            file_count += 1
            samples_written = 0
    if samples_written:
        tfrecord_writer.close()


def ds_from_paths(paths, batch_size):
    ds = tf.data.TFRecordDataset(filenames=paths)
    ds = ds.map(_parse_function).shuffle(buffer_size=100)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def read_tfrecords(folder_path, bs=16):
    fp = glob(os.path.join(folder_path, '*.tfrecord'))
    random.shuffle(fp)
    nt = len(fp) // 10
    return ds_from_paths(fp[nt:], bs), ds_from_paths(fp[:nt], bs)


def load_dataset(batch_size):
    return tf.data.Dataset.from_generator(
        lambda: data_gen("data/datasets_fullband/",
                         "data/MS-SNSD/", "data/rirs_noises/"),
        output_signature=(
            tf.TensorSpec(shape=(SAMPLES, NUM_BINS, 1), dtype=np.float32),
            tf.TensorSpec(shape=(SAMPLES, NUM_BINS, 1), dtype=np.float32),
        )
    ).batch(batch_size)


def _parse_function(proto):
    keys_to_features = {
        'signal': tf.io.FixedLenFeature([], tf.string),
        'noise': tf.io.FixedLenFeature([], tf.string)
    }
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    parsed_features['signal'] = tf.io.parse_tensor(
        parsed_features['signal'], out_type=tf.float32)
    parsed_features['noise'] = tf.io.parse_tensor(
        parsed_features['noise'], out_type=tf.float32)

    return parsed_features['signal'], parsed_features['noise']


if __name__ == "__main__":
    gen = data_gen("data/datasets_fullband/",
                   "data/MS-SNSD/", "data/rirs_noises/")
    write_tfrecords(f"data/records_{NUM_BINS}", gen)
    # for a, b in gen:
    #     print(a.shape, b.shape)
    #     break
