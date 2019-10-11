import os
import sys
import logging
import shutil
import tempfile
import json


def __norm_and_join(*args):
    return os.path.normpath(os.path.join(*args))


__flatbuffer_xmos_dir = __norm_and_join(
    os.path.dirname(os.path.realpath(__file__)), '..', 'flatbuffers_xmos')

DEFAULT_SCHEMA = __norm_and_join(__flatbuffer_xmos_dir, 'schema.fbs')

if sys.platform.startswith("linux"):
    DEFAULT_FLATC = __norm_and_join(__flatbuffer_xmos_dir, 'flatc_linux')
elif sys.platform == "darwin":
    DEFAULT_FLATC = __norm_and_join(__flatbuffer_xmos_dir, 'flatc_darwin')
else:
    DEFAULT_FLATC = shutil.which("flatc")


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
