import tensorflow as tf
import tensorflow.keras.layers as l
import numpy as np

TIMESTEPS = None
SAMPLES = 321
LOG10 = np.log(10)


def red_sum(t, kd=True):
    return tf.reduce_sum(t, axis=[1, 2], keepdims=kd)


def si_sdr_loss(y_true, y_pred):
    alpha = red_sum(y_true * y_pred) / red_sum(tf.square(y_true))
    s_target = alpha * y_true
    si_sdr = 10 * tf.math.log(red_sum(tf.square(s_target), False) / (
        red_sum(tf.square(s_target-y_pred), False) + 1e-7)) / LOG10
    return -si_sdr


def bn_relu(x):
    return l.ReLU()(l.BatchNormalization()(x))


def get_trunet():
    inp = l.Input(shape=(SAMPLES, 257, 1))
    x = tf.pad(inp, tf.constant([[0, 0], [0, 0], [0, 2], [0, 0]]))
    x = l.Conv2D(64, kernel_size=[1, 5], strides=[1, 2], padding="valid")(x)
    xs = [bn_relu(x)]
    for k, s in zip([3, 5, 3, 5, 3], [1, 2, 1, 2, 2]):
        # SEPARABLE!!!
        x = l.Conv2D(128, kernel_size=[1, k],
                     strides=[1, s], padding="same")(xs[-1])
        x = bn_relu(x)
        xs.append(x)
    x = tf.reshape(xs[-1], [-1, 16, 128])
    x = l.Bidirectional(l.GRU(64, return_sequences=True))(x)
    x = tf.reshape(x, [-1, SAMPLES, 16, 128])
    x = l.Conv1D(64, kernel_size=1, strides=1, padding="same")(x)
    x = bn_relu(x)
    x = tf.reshape(tf.transpose(x, [0, 2, 1, 3]), [-1, SAMPLES, 64])
    x = l.GRU(128, return_sequences=True)(x)
    x = tf.transpose(tf.reshape(x, [-1, 16, SAMPLES, 128]), [0, 2, 1, 3])
    x = l.Conv2D(128, kernel_size=[1, 1], strides=1, padding="same")(x)
    x = bn_relu(x)
    for k, s, skip in zip([3, 5, 3, 5, 3], [2, 2, 1, 2, 1], xs[:1:-1]):
        x = l.Concatenate()([x, skip])
        x = l.Conv2D(64, kernel_size=[1, 1], strides=1, padding="same")(x)
        x = bn_relu(x)
        # x = l.BatchNormalization()(x)
        x = l.Conv2DTranspose(
            64, kernel_size=[1, k], strides=[1, s], padding="same")(x)
        x = bn_relu(x)
    x = l.Concatenate()([x, xs[0]])
    x = l.Conv2DTranspose(1, kernel_size=[1, 5], strides=[1, 2],
                          padding="valid")(x)[:, :, :257]
    out = tf.exp(x)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    optimizer = tf.keras.optimizers.Adam(4e-4)
    model.compile(optimizer=optimizer, loss="mse")
    return model


if __name__ == "__main__":
    model = get_trunet()
    print(np.sum([np.prod(i.shape) for i in model.trainable_weights]))
    # tf.keras.utils.plot_model(model, show_shapes=True)
