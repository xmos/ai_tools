# tflite2xcore application

## Building

Install CMake version 3.11.1 or newer (https://cmake.org/download/).

Modify your path to include the CMake binaries.  Run the CMake application and
click Tools...How to Install For Command Line Use.

Install Bazel version 0.26.1 (https://bazel.build/)

Build tensorflow

    > ./configure
    > bazel build --config=v2 //tensorflow/tools/pip_package:build_pip_package

Make a directory for the build

    > mkdir build
    > cd build

Run cmake

    > cmake ../src/
    > sudo make install

## Running

Run the application 

    > tflite2xcore path/to/input_model.tflite path/to/output_model.tflite

Or, for help on options

    > ./tflite2xcore --help

## Using JSON Models

You can experiment with tflite model flatbuffer files using the JSON format.
The flatbuffer schema compiler (`flatc`) can be used to convert the binary format to 
JSON text.

To begin, build flatbuffers

    > mkdir build
    > cd build/
    > ccmake ../
    > cmake ../
    > make
    > sudo make install

To convert a tflite model flatbuffer file to JSON

    > flatc --json path/to/tensorflow/tensorflow/lite/schema/schema.fbs -- path/to/your/model.tflite

To convert a JSON text file to tflite model flatbuffer file

    > flatc --binary path/to/tensorflow/tensorflow/lite/schema/schema.fbs path/to/your/model.json

See https://google.github.io/flatbuffers/flatbuffers_guide_using_schema_compiler.html for more information on `flatc`.
