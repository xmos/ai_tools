import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (Input, GRU, ReLU, Reshape,
                                     BatchNormalization, Conv2D,
                                     Conv2DTranspose, Concatenate)

TIMESTEPS = None
SAMPLES = 1281


def bn_relu(x):
    return ReLU()(BatchNormalization()(x))


def simple_sigmoid(x):
    return tf.clip_by_value(ReLU()(x + .5), 0, 1)


def simple_tanh(x):
    return tf.clip_by_value(x, -1, 1)


def gru_block(x, num_samples, enc_f, state_input):
    num_chans = 32
    x = Reshape([num_samples, num_chans * enc_f])(x)
    if state_input is not None:
        x, state = GRU(num_chans * enc_f, return_state=True,
                       return_sequences=True, unroll=True)(x, state_input)
        # activation=simple_tanh,
        # recurrent_activation=simple_sigmoid)(x, state_input)
    else:
        x, state = GRU(num_chans * enc_f, return_sequences=True)(x), None
        # activation=simple_tanh,
        # recurrent_activation=simple_sigmoid)(x), None
    x = Reshape([num_samples, enc_f, 32])(x)
    return x, state


def get_trunet(num_freqs=64, num_samples=SAMPLES, inference=False):
    channels = [24, 32, 48, 48, 64, 64]
    strides = [2, 1, 2, 1, 2, 2]
    k_sizes = [5, 3, 5, 3, 5, 3]
    zipped = list(zip(k_sizes, strides, channels))
    inp = Input(shape=(num_samples, num_freqs, 1))
    state_input = Input(shape=(64,)) if inference else None
    x = BatchNormalization()(inp)
    x = Conv2D(channels[0], kernel_size=[1, k_sizes[0]],
               strides=[1, strides[0]], padding="same", use_bias=False)(x)
    x = bn_relu(x)
    xs = [x]
    for k, s, c in zipped[1:]:
        x = Conv2D(c, kernel_size=[1, k], strides=[
                   1, s], padding="same", use_bias=False)(x)
        x = bn_relu(x)
        xs.append(x)
    x = Conv2D(32, kernel_size=[1, 2], strides=[1, 2],
               padding="same", use_bias=False)(x)
    x = bn_relu(x)
    x, new_state = gru_block(x, num_samples, 2, state_input)
    x = Conv2DTranspose(32, kernel_size=[1, 2], strides=[1, 2],
                        padding="same", use_bias=False)(x)
    x = bn_relu(x)
    for (k, s, c), skip in list(zip(zipped, xs))[:1:-1]:
        cs = (c * 2) // 3
        x = Concatenate()([x, skip])
        x = Conv2D(cs, kernel_size=[1, 1], use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(cs, kernel_size=[1, k], strides=[1, s],
                            padding="same", use_bias=False)(x)
        x = bn_relu(x)
    x = Concatenate()([x, xs[0]])
    x = Conv2DTranspose(1, kernel_size=[1, k_sizes[0]],
                        strides=[1, strides[0]], padding="same")(x)
    out = tf.keras.activations.sigmoid(x)
    inputs = [inp, state_input] if inference else inp
    outputs = [out, new_state] if inference else out * inp
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    model = get_trunet(64, 1, inference=True)
    print(np.sum([np.prod(i.shape) for i in model.trainable_weights]))
    tf.keras.utils.plot_model(model, show_shapes=True)
