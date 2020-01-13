# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import argparse
import logging
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import model_interface as mi
import tflite_utils

DEFAULT_INPUTS = 32
DEFAULT_OUTPUTS = 16
DEFAULT_K_H = 3
DEFAULT_K_W = DEFAULT_K_H
DEFAULT_HEIGHT = 5
DEFAULT_WIDTH = DEFAULT_HEIGHT


# Prepare data function
def generate_data(height, width, inputs):
    quant_data = np.float32(
        np.random.uniform(0, 1, size=(10, height, width, inputs)))
    x_test_float = np.concatenate(
        [np.zeros((1, height, width, inputs), dtype=np.float32),
         quant_data[:3, :, :, :]],  # pylint: disable=unsubscriptable-object
        axis=0)
    return x_test_float, quant_data


# Class for the model
class Conv2dDeepinDeepoutRelu(mi.KerasModel):
    def build(self, K_h, K_w, height, width):
        input_dim = self.input_dim
        output_dim = self.output_dim
        # Env
        tf.keras.backend.clear_session()
        tflite_utils.set_all_seeds()
        # Building
        model = tf.keras.Sequential(name=self.name)
        model.add(layers.Conv2D(
            filters=output_dim,
            kernel_size=(K_h, K_w),
            padding='same',
            input_shape=(height, width, input_dim)))
        # Compilation
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        # Add to dict
        self.models[self.name] = model
        # Show summary
        model.summary()
        return model

    def train():  # Not training this model
        pass

    # For training
    def prep_data(self, height, width):
        self.data['export_data'], self.data['quant'] = generate_data(
            height, width, self.input_dim)

    # For exports
    def gen_test_data(self, height, width):
        if not self.data:
            self.prep_data(height, width)


def main(input_dim=DEFAULT_INPUTS,
         output_dim=DEFAULT_OUTPUTS,
         K_h=DEFAULT_K_H,
         K_w=DEFAULT_K_W,
         height=DEFAULT_HEIGHT,
         width=DEFAULT_WIDTH):
    assert input_dim % 32 == 0, "# of input channels must be multiple of 32"
    assert output_dim % 16 == 0, "# of output channels must be multiple of 16"
    # Instantiate model
    test_model = Conv2dDeepinDeepoutRelu(
        'conv2d_deepin_deepout_relu', Path('.'), input_dim, output_dim)
    # Build model and compile
    test_model.build(K_h, K_w, height, width)
    # Generate test data
    test_model.gen_test_data(height, width)
    # Populate converters
    # test_model.populate_converters() #breaks in xcore
    test_model.to_tf_float()
    test_model.to_tf_quant()
    test_model.to_tf_stripped()
    # Convert and save, will break in stripped
    test_model.convert_and_save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--use_gpu', action='store_true', default=False,
        help='Use GPU for training. Might result in non-reproducible results')
    parser.add_argument('--inputs', type=int, default=DEFAULT_INPUTS,
                        help='Number of input channels')
    parser.add_argument('--outputs', type=int, default=DEFAULT_OUTPUTS,
                        help='Number of output channels')
    parser.add_argument('--height', type=int, default=DEFAULT_HEIGHT,
                        help='Height of input image')
    parser.add_argument('--width', type=int, default=DEFAULT_WIDTH,
                        help='Width of input image')
    parser.add_argument('--K_h', type=int, default=DEFAULT_K_H,
                        help='Height of kernel')
    parser.add_argument('--K_w', type=int, default=DEFAULT_K_W,
                        help='Width of kernel')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Verbose mode.')
    args = parser.parse_args()

    verbose = args.verbose

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.getLogger('tensorflow').setLevel(logging.ERROR)

    logging.info(f"Eager execution enabled: {tf.executing_eagerly()}")

    tflite_utils.set_gpu_usage(args.use_gpu, verbose)

    main(input_dim=args.inputs, output_dim=args.outputs,
         K_h=args.K_h, K_w=args.K_w,
         height=args.height, width=args.width)