# XCORE Flexbuffer Python Bindings

This project contains Python bindings for a subset of the Flexbuffer classes.  This subset is used to serialize model custom options using the Flexbuffer format.

## Building

Install CMake version 3.14 or newer (https://cmake.org/download/).

Modify your path to include the CMake binaries.  Run the CMake application and
click Tools...How to Install For Command Line Use.

Make a directory for the build

    > mkdir build
    > cd build

Run cmake

    > cmake ../
    > make

To install

    > make install
