FROM tensorflow/build:2.15-python3.11
RUN pip install cmake==3.28.3 setuptools~=70.0 setuptools-scm
