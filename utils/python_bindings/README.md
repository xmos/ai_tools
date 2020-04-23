# Python Flexbuffer helpers

## Prerequisites

### macOS

The minimum macOS version for tflite2xcore is 10.13 (High Sierra) or newer.  To 
build for 10.13, you may need to install MacOSX10.13.sdk.  It can be downloaded from:

https://github.com/phracker/MacOSX-SDKs

Place the SDK folder in: 

/Library/Developer/CommandLineTools/SDKs/


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