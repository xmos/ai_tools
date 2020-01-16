# Copyright (c) 2019, XMOS Ltd, All rights reserved
import sys
import setuptools

if sys.platform.startswith("linux"):
    LIB_TFLITE2XCORE = 'serialization/linux/libtflite2xcore.so.1.0.1'
elif sys.platform == "darwin":
    LIB_TFLITE2XCORE = 'serialization/macos/libtflite2xcore.1.0.1.dylib'
else:
    LIB_TFLITE2XCORE = 'serialization/windows/libtflite2xcore.dll'

setuptools.setup(
    name='tflite2xcore',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
    ],
    package_data={'': [LIB_TFLITE2XCORE]},
)
