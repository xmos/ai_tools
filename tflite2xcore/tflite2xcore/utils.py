# Copyright (c) 2018-2020, XMOS Ltd, All rights reserved

import os
import random
import logging

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import warnings
warnings.filterwarnings(action='ignore')
import tensorflow as tf  # nopep8
from tensorflow import keras
warnings.filterwarnings(action='default')


VE, ACC_PERIOD, WORD_SIZE = 32, 16, 4
DEFAULT_SEED = 123


def set_all_seeds(seed=DEFAULT_SEED):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_gpu_usage(use_gpu, verbose):
    # can throw annoying error if CUDA cannot be initialized
    default_log_level = os.environ['TF_CPP_MIN_LOG_LEVEL']
    if not verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = default_log_level

    if gpus:
        if use_gpu:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, enable=True)
        else:
            logging.info("GPUs disabled.")
            tf.config.experimental.set_visible_devices([], 'GPU')
    elif use_gpu:
        logging.warning('No available GPUs found, defaulting to CPU.')
    logging.debug(f"Eager execution enabled: {tf.executing_eagerly()}")


def set_verbosity(verbose=False):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.getLogger('tensorflow').setLevel(logging.ERROR)


class LoggingContext():
    def __init__(self, logger, level=None, handler=None, close=True):
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, exception_type, exception_value, traceback):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()


class Log():
    @classmethod
    def _array_msg(cls, arr, style=''):
        msg = f"numpy.ndarray (shape={arr.shape}, dtype={arr.dtype}):\n"
        if style.endswith('_shift_scale_arr'):
            msg += f"shift_pre:\n{arr[:, 0]}\n"
            msg += f"scale:\n{arr[:, 1]}\n"
            msg += f"shift_post:\n{arr[:, 2]}"
        else:
            msg += f"{arr}"
        return msg + '\n'
