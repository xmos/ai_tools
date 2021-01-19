# Copyright (c) 2019, XMOS Ltd, All rights reserved
import setuptools

LIB_FLEXBUFFERS = [
    "libs/linux/libflexbuffers.so",
    "libs/linux/libflexbuffers.so.1.0.1",
    "libs/macos/libflexbuffers.dylib",
    "libs/macos/libflexbuffers.1.0.1.dylib",
]

EXCLUDES = ["*tests", "*tests.*", "*model_generation", "*model_generation.*"]
SCRIPTS = ["xformer.py", "tflite2xcore/tflite_visualize.py"]

INSTALL_REQUIRES = [
    "aenum==2.2.4",
    "dill==0.3.1.1",
    "flatbuffers==1.12.0",
    "matplotlib==3.1.1",
    "numpy==1.19.2",
    "tensorflow==2.3.0",
    "typing-extensions==3.7.4",
]

setuptools.setup(
    name="tflite2xcore",
    packages=setuptools.find_packages(exclude=EXCLUDES),
    scripts=SCRIPTS,
    python_requires=">=3.6.8",
    install_requires=INSTALL_REQUIRES,
    package_data={"": LIB_FLEXBUFFERS},
    author="XMOS",
    author_email="support@xmos.com",
    description="XMOS Tools to convert TensorFlow Lite models to xCORE microcontrollers.",
    license="LICENSE.txt",
    keywords="xmos xcore",
    use_scm_version={
        "root": "..",
        "relative_to": __file__,
        "version_scheme": "post-release",
    },
    setup_requires=["setuptools_scm"],
)
