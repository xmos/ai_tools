#!/usr/bin/env python -O
# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

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
        "--remove_input_alignment_pad",
        action="store_true",
        default=False,
        help="Remove channel-wise padding on the input tensor(s). "
        "The new input tensor will have the padded size, "
        "so the padding should be implemented by the application developer.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=version.get_version(),
        help="Display the version of the xformer",
    )
    parser.add_argument(
        "--ext_mem",
        action="store_true",
        default=False,
        help="Experimental flag for better external memory support.",
    )
    parser.add_argument(
        "--experimental-xformer2",
        action="store_true",
        default=False,
        help="Use MLIR-based xformer 2.0 for part of the optimization pipeline. Experimental.",
    )
    parser.add_argument(
        "--only-experimental-xformer2",
        action="store_true",
        default=False,
        help="Use only MLIR-based xformer 2.0. Experimental.",
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
        remove_input_alignment_pad=args.remove_input_alignment_pad,
        external_memory=args.ext_mem,
        experimental_xformer2=args.experimental_xformer2,
        only_experimental_xformer2=args.only_experimental_xformer2
    )

    print(f"Conversion successful, output: {tflite_output_path}")

    if args.analyze:
        analyze.print_report(tflite_output_path)
