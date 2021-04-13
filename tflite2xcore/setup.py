# Copyright 2019-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
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
    "aenum>=2.2.4",
    "dill>=0.3.1.1",
    "flatbuffers==1.12.0",
    "numpy>=1.19.5",
    "tensorflow>=2.4.0,<=2.4.1",
    "larq-compute-engine>=0.5.0",
]

setuptools.setup(
    name="tflite2xcore",
    packages=setuptools.find_packages(exclude=EXCLUDES),
    scripts=SCRIPTS,
    python_requires=">=3.8.0",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "test": [
            "pytest>=5.2.0",
            "pytest-xdist>=1.30.0",
            "portalocker==2.0.0",
            "keras-applications>=1.0.8",
            "PyYAML>=5.3.1",
            "larq>=0.11.1",
        ],
        "examples": [
            "scipy>=1.4.1",
            "keras-preprocessing>=1.1.2",
            "tqdm>=4.41.1",
            "matplotlib>=3.1.1",
            "jupyter>=1.0.0",
        ],
        "dev": [
            "mypy>=0.782",
            "black>=19.10b0",
            "pylint>=2.4.2",
            "lhsmdu>=1.1",
        ],
    },
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
