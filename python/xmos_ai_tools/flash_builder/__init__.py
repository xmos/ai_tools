#!/usr/bin/env python
# Copyright (c) 2020, XMOS Ltd, All rights reserved

import sys
import argparse
from .flash import FlashBuilder

# parser = argparse.ArgumentParser(description='Build parameter/flash images')
# parser.add_argument('--output', default='image.bin',  help='output file')
# parser.add_argument('--target', default='host',       help='"flash" or "host" (default)')
# parser.add_argument('files',    nargs='+', help='Model and parameter files, - indicates a missing one, must be an even number of files for "flash" (model params model params ...), or a single file for "host" (params)')

# args = parser.parse_args()

# if args.target == 'flash' or args.target == 'xcore':
#     if len(args.files) %2 != 0:
#         parser.print_usage()
#         sys.exit(1)
#     engines = len(args.files)//2
#     fb = FlashBuilder(engines)
#     for i in range(engines):
#         fb.add_model(i, filename = args.files[2*i])
#         fb.add_params(i, filename = args.files[2*i+1])

#     fb.flash_file(args.output)

# elif args.target == 'host':
#     if len(args.files) != 1:
#         parser.print_usage()
#         sys.exit(1)
#     output = FlashBuilder.create_params_file(args.output, input_filename = args.files[0])

# else:
#     parser.print_usage()
#     sys.exit(1)

def generate_flash(model_file, params_file, output_file):
    fb = FlashBuilder()
    fb.add_model(0, filename=model_file)
    fb.add_params(0, filename=params_file)
    fb.flash_file(output_file)

