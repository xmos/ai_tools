import time
import os
import sys
import itertools
import re
from glob import glob

import openai

from xmos_ai_tools import xformer as xf

def get_mlir(model_path, after=False):
    out_file = "model.mlir"
    params = {"mlir-print-ir-after-all": None}
    if not after:
        params["lce-translate-tfl"] = None
    xf.convert(model_path, out_file, params=params) # type: ignore
    os.remove(out_file)
    model = xf.__compilation_output  # type: ignore
    patterns = [r'dense<"0x(.*?)"', r'bytes : "0x(.*?)"']
    for pattern in itertools.chain(*[re.findall(p, model) for p in patterns]):
        model = model.replace(pattern, f"[PLACEHOLDER: {len(pattern) // 2} bytes]")
    idx = [m.start() for m in re.finditer("\n", model)]
    model = model[:idx[-2]]
    return model

model_directories = glob("../models/8x8/*")
for dir in model_directories:
    models = glob(dir + "/*.tflite")
    for model in models:
        test_subject = model.split("/")[-2]
        print("Test subject:")
        print(test_subject)
        mlir = get_mlir(model)
        print("MLIR:")
        print(mlir)
    # print(models)
    break
