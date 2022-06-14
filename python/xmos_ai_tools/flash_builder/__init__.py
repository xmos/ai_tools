#!/usr/bin/env python
# Copyright (c) 2020, XMOS Ltd, All rights reserved

import sys
import argparse
from .flash import FlashBuilder

def generate_flash(model_file, params_file, output_file):
    fb = FlashBuilder()
    fb.add_model(0, filename=model_file)
    fb.add_params(0, filename=params_file)
    fb.flash_file(output_file)

