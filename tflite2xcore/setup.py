# Copyright (c) 2019, XMOS Ltd, All rights reserved
import setuptools

LIB_TFLITE2XCORE = [
    "libs/linux/libtflite2xcore.so",
    "libs/linux/libtflite2xcore.so.1.0.1",
    "libs/macos/libtflite2xcore.dylib",
    "libs/macos/libtflite2xcore.1.0.1.dylib",
]

EXCLUDES = ["*tests", "*tests.*", "*model_generation", "*model_generation.*"]
SCRIPTS = ["xformer.py", "tflite2xcore/tflite_visualize.py"]

INSTALL_REQUIRES = [
    "numpy>=1.17.2",
    "flatbuffers==1.12.0",
    "tensorflow>=2.3.0",
    "aenum>=2.2.4",
    "matplotlib>=3.1.1",
]

setuptools.setup(
    name="tflite2xcore",
    version="0.1.1",
    packages=setuptools.find_packages(exclude=EXCLUDES),
    scripts=SCRIPTS,
    python_requires=">=3.6.8",
    install_requires=INSTALL_REQUIRES,
    package_data={"": LIB_TFLITE2XCORE},
    author="XMOS",
    author_email="support@xmos.com",
    description="XMOS Tools to convert TensorFlow Lite models to xCORE microcontrollers.",
    license="LICENSE.txt",
    keywords="xmos xcore",
)
