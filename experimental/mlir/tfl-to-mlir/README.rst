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

This code walks through a network and prints the MLIR, here are two alternative
approaches to printing that might work better if the missing components are found:

  // OwningModuleRef attempt -- missing OpAsmPrinter
  // OpAsmPrinter p();
  // mod.get().print(p);

  // op builder attempt -- need to get model into builder
  // OpBuilder builder(&context);
  // for( Block & block : *builder.getBlock())
  //   printBlock(block);

