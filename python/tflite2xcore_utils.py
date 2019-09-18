import os
import shutil
import tempfile
import json


DEFAULT_SCHEMA = os.path.normpath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'schema.fbs'
))


DEFAULT_FLATC = shutil.which("flatc")


def load_tflite_as_json(tflite_input, *,
                        flatc_bin=DEFAULT_FLATC, schema=DEFAULT_SCHEMA):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # convert model to json
        cmd = ("{flatc_bin} -t --strict-json --defaults-json "
            "-o {tmp_dir} {schema} -- {input}").format(
            flatc_bin=flatc_bin, input=tflite_input,
            schema=schema, tmp_dir=tmp_dir)
        print(cmd)
        os.system(cmd)

        # open json file
        json_input = os.path.join(tmp_dir,
            os.path.splitext(os.path.basename(tflite_input))[0] + ".json")
        with open(json_input, 'r') as f:
            model = json.load(f)

    return model


def save_json_as_tflite(model, tflite_output, *,
                        flatc_bin=DEFAULT_FLATC, schema=DEFAULT_SCHEMA):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # write new json file
        json_output = os.path.join(tmp_dir,
            os.path.splitext(os.path.basename(tflite_output))[0] + ".tmp.json")
        with open(json_output, 'w') as f:
            json.dump(model, f, indent=2)

        # convert to tflite
        cmd = ("{flatc_bin} -b --strict-json --defaults-json "
            "-o {tmp_dir} {schema} {json_output}").format(
            flatc_bin=flatc_bin, json_output=json_output,
            schema=schema, tmp_dir=tmp_dir)
        print(cmd)
        os.system(cmd)

        # move to specified location
        tmp_tflite_output = os.path.join(tmp_dir,
            os.path.splitext(os.path.basename(tflite_output))[0] + ".tmp.tflite")
        shutil.move(tmp_tflite_output, tflite_output)