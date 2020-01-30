import tensorflow as tf
import numpy as np
import initializers as init


# Class with custom initializers
class Linear(tf.keras.layers.Layer):
    def __init__(self, units=32, input_dim=32, mu=0, sigma=0.05, seed=42):
        super(Linear, self).__init__()
        w_init = init.create_initializer(
            'random_normal',conf={'mean':mu,'stddev':sigma, 'seed': seed})

        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units),
                                                  dtype='float32'),
                             trainable=True)
        b_init = init.create_initializer(
            'random_normal',conf={'mean':mu,'stddev':sigma, 'seed': seed})
        self.b = tf.Variable(initial_value=b_init(shape=(units,),
                                                  dtype='float32'),
                             trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b



mu = -50
sigma = 0.25
seed = 42
x = tf.ones((2,2))
linear_layer = Linear(4, 2, mu, sigma, seed)
y = linear_layer(x)
print(y)

w_init = init.create_initializer(
    'random_normal',conf={'mean':mu,'stddev':sigma, 'seed': seed})
b_init = init.create_initializer(
    'random_normal',conf={'mean':mu,'stddev':sigma, 'seed': seed})
sequential_conv = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
                           input_shape=(None,2,2),
                           filters=1,
                           kernel_size=(1, 1),
                           strides=(1, 1),
                           padding='valid',
                           dilation_rate=(1,1),
                           activation=None,
                           use_bias=True,
                           kernel_initializer=w_init,
                           bias_initializer=b_init
    )
])


y=tf.reshape(x, [1, 1, 2, 2])
sequential_conv.build()
sequential_conv.summary()
z = sequential_conv(y)
print(z)
