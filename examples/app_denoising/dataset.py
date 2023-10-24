import os
import tensorflow as tf
import random
from tqdm import tqdm
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, resample
import pyroomacoustics as pra
from glob import glob

SAMPLES = None
WINDOW_SIZE = 512
HOP_SIZE = 128
FFT_SIZE = 512
FL = 512 * 8 * 10  # * 5
SR = 16000
NUM_BINS = 257
tf.keras.utils.set_random_seed(42)


def sample():
    a = np.random.lognormal(mean=-2.5, sigma=0.8)
    return a if (a >= 0.01 and a <= 0.8) else sample()


def generate_random_params():
    room_dim = np.random.uniform(3, 8, size=3)
    return {
        'absorb': sample(),
        'room_dim': room_dim,
        'src_loc': np.random.uniform(1, room_dim - .5, size=3),
        'mic_loc': np.random.uniform(1, room_dim - .5, size=3),
        'snr': np.random.uniform(10, 30)
    }


def nsf(signal, noise, snr):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    return np.sqrt((signal_power / noise_power) * 10 ** (-snr / 10.0))


def sim_room(signal, params, noise=None, reverb=True):
    room = pra.ShoeBox(
        params['room_dim'], fs=SR, materials=pra.Material(
            params["absorb"] if reverb else 1.),
        use_rand_ism=True, max_rand_disp=0.05
    )
    if noise is not None:
        noise = noise * nsf(signal, noise, params["snr"])
        signal += noise
    room.add_source(params['src_loc'], signal=signal)
    room.add_microphone(loc=params['mic_loc'])
    room.simulate()
    reverb_audio_data = room.mic_array.signals[0]
    return reverb_audio_data[:len(signal)]


def read_file(path):
    fs, a = wavfile.read(path)
    return a


def process_signal(signal, noise):
    p = generate_random_params()
    c = sim_room(signal, p, None, False)
    r = sim_room(signal, p, noise, True) - c
    return c, r


def apply_stft(signal):
    _, _, s = stft(signal, fs=SR, nperseg=WINDOW_SIZE,
                   noverlap=WINDOW_SIZE - HOP_SIZE, nfft=FFT_SIZE)
    return s.T


def get_input(signal, noise, return_phase=False):
    c, r = process_signal(signal, noise)
    mix_spec = apply_stft(c+r)
    clean_spec = apply_stft(c)
    clean_mag = np.log1p(np.abs(clean_spec))
    mix_mag = np.log1p(np.abs(mix_spec))
    a = mix_mag[..., None].astype(np.float32)
    b = clean_mag[..., None].astype(np.float32)
    if return_phase:
        return a, b, np.angle(mix_spec), c
    return a, b


def signal_gen(folder, res=True):
    paths = glob(f"{folder}/**/*.wav", recursive=True)
    random.shuffle(paths)
    for path in paths:
        s = read_file(path)
        s = resample(s, len(s) // (3 if res else 1))
        yield from (s[i-FL:i] for i in range(FL, len(s), FL))


def noise_gen(folder):
    while True:
        yield from signal_gen(folder, False)


def data_gen(sig_fol, noise_fol, phase=False):
    gen = zip(signal_gen(sig_fol), noise_gen(noise_fol))
    yield from (get_input(s, n, phase) for s, n in gen)


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
    for signal, noise in tqdm(data_generator, total=2326):
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


def read_tfrecords(folder_path, batch_size=16):
    file_paths = glob(os.path.join(folder_path, '*.tfrecord'))
    random.shuffle(file_paths)
    num_test = len(file_paths) // 10
    test_ds = tf.data.TFRecordDataset(filenames=file_paths[:num_test])
    test_ds = test_ds.map(_parse_function).batch(batch_size)
    train_ds = tf.data.TFRecordDataset(filenames=file_paths[num_test:])
    train_ds = train_ds.map(_parse_function).batch(batch_size)
    return train_ds, test_ds


def load_dataset(batch_size):
    return tf.data.Dataset.from_generator(
        lambda: data_gen("data/datasets_fullband/", "data/MS-SNSD/"),
        output_signature=(
            tf.TensorSpec(shape=(SAMPLES, 257, 1), dtype=np.float32),
            tf.TensorSpec(shape=(SAMPLES, 257, 1), dtype=np.float32),
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
    gen = data_gen("data/datasets_fullband/", "data/MS-SNSD/")
    write_tfrecords("data/records", gen)
    # for a, b, in gen:
    #     print(a.shape, b.shape)
    #     break
