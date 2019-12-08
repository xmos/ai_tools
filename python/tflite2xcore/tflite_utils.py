import os
import sys
import logging
import shutil
import tempfile
import json
import random

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import warnings
warnings.filterwarnings(action='ignore')
import tensorflow as tf  # nopep8
warnings.filterwarnings(action='default')




def __norm_and_join(*args):
    return os.path.normpath(os.path.join(*args))


__flatbuffer_xmos_dir = __norm_and_join(
    os.path.dirname(os.path.realpath(__file__)),
    '..', '..', 'third_party', 'flatbuffers')

from serialization.flatbuffers_io import DEFAULT_SCHEMA

#DEFAULT_SCHEMA = __norm_and_join(__flatbuffer_xmos_dir, 'schema.fbs')

"""if sys.platform.startswith("linux"):
    DEFAULT_FLATC = __norm_and_join(__flatbuffer_xmos_dir, 'flatc_linux')
elif sys.platform == "darwin":
    DEFAULT_FLATC = __norm_and_join(__flatbuffer_xmos_dir, 'flatc_darwin')
else:"""
# TODO: this needs to be fixed
DEFAULT_FLATC = shutil.which("flatc")


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


def check_schema_path(schema):
    if not os.path.exists(schema):
        raise FileNotFoundError(
            "Sorry, schema file cannot be found at {}".format(schema))


def check_flatc_path(flatc):
    if flatc is None:
        raise RuntimeError("Sorry, cannot find flatc")
    elif not os.path.exists(flatc):
        raise RuntimeError(
            "Sorry, flatc is not available at {}".format(flatc))


def load_tflite_as_json(tflite_input, *,
                        flatc_bin=DEFAULT_FLATC, schema=DEFAULT_SCHEMA):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # convert model to json
        cmd = (f"{flatc_bin} -t --strict-json --defaults-json "
               f"-o {tmp_dir} {schema} -- {tflite_input}")
        logging.info(f"Executing: {cmd}")
        os.system(cmd)

        # open json file
        json_input = os.path.join(
            tmp_dir,
            os.path.splitext(os.path.basename(tflite_input))[0] + ".json")
        with open(json_input, 'r') as f:
            model = json.load(f)

    return model


def save_json_as_tflite(model, tflite_output, *,
                        flatc_bin=DEFAULT_FLATC, schema=DEFAULT_SCHEMA):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # write new json file
        json_output = os.path.join(
            tmp_dir,
            os.path.splitext(os.path.basename(tflite_output))[0] + ".tmp.json")
        with open(json_output, 'w') as f:
            json.dump(model, f, indent=2)

        # convert to tflite
        cmd = (f"{flatc_bin} -b --strict-json --defaults-json "
               f"-o {tmp_dir} {schema} {json_output}")
        logging.info(f"Executing: {cmd}")
        os.system(cmd)

        # move to specified location
        tmp_tflite_output = os.path.join(
            tmp_dir,
            os.path.splitext(os.path.basename(tflite_output))[0] + ".tmp.tflite")
        shutil.move(tmp_tflite_output, tflite_output)


class LoggingContext:
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

    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()
        # implicit return of None => don't swallow exceptions
