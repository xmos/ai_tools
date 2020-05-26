# Scope

This document provides a high level description of the tflite2xcore module concepts.  The goal of this document is to orient a new developer to the concepts implement in the module.  

## Overview

The graph transformer is responsible for converting a quantized TensorFlow Lite model, or computational graph, into a format that is optimized for VPU-enabled xCORE microcontrollers.  The transformer is designed to be similar to an LLVM or MLIR optimizing compiler.  The model graph is transformed by a series of transformation passes - each one potentially making edits to the model graph.  Each transformation pass looks for patterns in the graph and, if a pattern is matched, the transformation pass applies various edits to the graph.  When all passes have run, the ruslting model is optimixed and ready to be deployed to the xCORE.

## Principal Components

### xCORE Model

The XCORE Model is a data structure that represents a mutable TensorFlow Lite Micro model.

### xCORE Interpreter

The XCORE Interpreter (see xcore_interpreter.py) is a TensorFlow Lite Micro runtime interpreter that can be executed on x86 platforms.  It supports inference on any TensorFlow Lite Micro model - including models converted to xCORE.  It supports this by calling the C versions of any optimized xCORE neural network kernel functions.

There are not native TensorFlow Lite Micro runtime interpreter Python bindings.  SO, the tflite2xcore model provides them via a thin C wrapper callable using `ctypes`.  This wrapper is implement in libtflite2xcore located in the `libs` directory. The soruce code libtflite2xcore is located under `utils\python_bindings` in the root of the AI tools repository.

### Graph Transformation

See [Graph Transformer](graph_transformer.md)

### Model Generation

See [Model Generation](model_generation.md)

### Model Serialization

See [Model Serialization](model_serialization.md)
