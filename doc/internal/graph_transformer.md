# Scope

This document provides a high level description of the graph transformer concepts.  The goal of this document is to orient a new developer to the concepts implement in the transformer.  Concepts from the tflite2xcore module are described elsewhere.

## Overview

The graph transformer is responsible for converting a quantized TensorFlow Lite model, or computational graph, into a format that is optimized for VPU-enabled xCORE microcontrollers.  The transformer is designed to be similar to an LLVM or MLIR optimizing compiler.  The model graph is transformed by a series of transformation passes - each one potentially making edits to the model graph.  Each transformation pass looks for patterns in the graph and, if a pattern is matched, the transformation pass applies various edits to the graph.  When all passes have run, the ruslting model is optimixed and ready to be deployed to the xCORE.

## Principal Components

### xformer

The xformer (see xformer.py) is the application script that an end user runs to perform a graph transformation.  The user specifies the input file, output file, various options to control the transformation, and other debugging options.

For more information on the xformer options, run:

    > xformer.py --help

### Converter

The converter (see converter.py) is a utility that executes the graph transformation.  It is called by the xformer, but can also be called from other Python components that wish to transform a model - including an end user's Jupyter notebook.  The main entry point for any conversion is `convert`.  This is where the input .tflite flatbuffer file is read into a [Model](tflite2xcore.md) object, the model is converted, and the output saved to a new .tflite flatbuffer file.

The model conversion involves registering all transformation passes with a pass manager, instructing the pass manager to run the passes, and performing a sanity check on the transformed model

### Pass Manager

The pass manager (see pass_manager.py) maintains a list of registered transformation passes and runs those passes.  For debugging purposes, the pass manager can also save intermediate outputs - including visualizations.

### Transformation Passes

A full list and description of all transformation passes is beyond the scope of this document.  Here is a partial list to give a flavor of the various passes:

- ReplaceReLUPass
- ReplaceDepthwiseConv2dPass
- ReplaceMaxPool2DPass
- ReplaceFullyConnectedPass
- LegalizeXCLookupTablePass
- FuseConv2dPaddingPass
- ParallelizeXCConv2dPass
- MinifyTensorNamesPass

Explore the `transformation_passes` folder for the most recent list of passes.  For more detail on any specific pass, read the source code for that pass and run the xformer in debug mode.

### Parallelization Planner

Several transformation passes are for optimizing an operator for parallel inference.  The parallelization planner constructs plan by splitting an output tensor by rows, cols, and/or channels.  At runtime, the par plan instructs the inference engine how to split the computation across cores.

## Testing

See the `tests\transformation_passes` and `tests\parallelization_plans` directories for graph tranformation unit tests.
