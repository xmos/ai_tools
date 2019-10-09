# tflite2xcore application

## Building

Install CMake version 3.11.1 or newer (https://cmake.org/download/).

Modify your path to include the CMake binaries.  Run the CMake application and
click Tools...How to Install For Command Line Use.

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
