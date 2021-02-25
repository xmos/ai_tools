Experimental TFLite to MLIR converter
=====================================

This is a simple program that opens a .tflite file, converts the 
network to MLIR, prints out a description of the network and then
converts the MLIR back to a flatbuffer and writes a .tflite file.

Bazel is used to build this program. The choice of build system was
driven by the most complex dependency (Tensorflow). If you have Bazel
installed (check .bazelversion file at top level for current version)
you can build with the following command:

    bazel build //experimental/mlir/tfl-to-mlir:tfl-to-mlir

To run the example app:

    ../../bazel-bin/experimental/mlir/tfl-to-mlir/tfl-to-mlir <intput.tflite>

@todo add test flatbuffer to repo
