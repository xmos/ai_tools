#!/usr/bin/env python -O
#
# Copyright (c) 2019, XMOS Ltd, All rights reserved

from pathlib import Path

from tflite2xcore import utils, analyze, version
import tflite2xcore.converter as xcore_conv

if __name__ == "__main__":
    parser = utils.VerbosityParser()
    parser.add_argument("tflite_input", help="Input .tflite file.")
    parser.add_argument("tflite_output", help="Output .tflite file.")
    parser.add_argument(
        "--minify",
        action="store_true",
        default=False,
        help="Make the model smaller at the expense of readability.",
    )
    parser.add_argument(
        "-par",
        "--num_threads",
        type=int,
        default=1,
        help="Number of parallel threads for xcore.ai optimization.",
    )
    parser.add_argument(
        "--intermediates_path",
        default=None,
        help="Path to directory for storing intermediate models. If not given "
        "intermediate models will not be saved. If path doesn't exists, "
        "it will be created. Contents may be overwritten.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        default=False,
        help="Analyze the output model. "
        "A report is printed showing the runtime memory footprint of the model.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=version.get_version(),
        help="Display the version of the xformer",
    )

    args = parser.parse_args()

    utils.set_verbosity(args.verbose)

    tflite_input_path = Path(args.tflite_input)
    tflite_output_path = Path(args.tflite_output)

    xcore_conv.convert(
        tflite_input_path,
        tflite_output_path,
        minification=args.minify,
        num_threads=args.num_threads,
        intermediates_path=args.intermediates_path,
    )

    print(f"Conversion successful, output: {tflite_output_path}")

    if args.analyze:
        analyze.print_report(tflite_output_path)
