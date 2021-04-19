FROM continuumio/miniconda3:4.8.2

# This Dockerfile is for use by the XMOS CI system
# It provides a minimal environment needed to execute the Jenkinsfile
# Most of the dependecies here are handled conda so we only include:
#  - conda setup
#  - xmos tools setup

# fix conda perms
RUN chmod -R 777 /opt/conda \
    && mkdir -p /.conda \
    && chmod -R 777 /.conda

# install tools lib dependencies
RUN apt-get update && apt-get install -y \
    libncurses5 libncurses5-dev \
    tcl environment-modules \
    && apt-get clean autoclean

# install compiler
RUN apt-get install -y build-essential

# install bazel
RUN apt-get install gnupg
ADD https://bazel.build/bazel-release.pub.gpg /tmp/
RUN cat /tmp/bazel-release.pub.gpg | gpg --dearmor > /etc/apt/trusted.gpg.d/
RUN echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list

# install get_tools.py script
#   requires connection to XMOS network at build and run time
#   if not possible, find another way to install the tools
RUN mkdir -m 777 /XMOS
ADD https://github0.xmos.com/raw/xmos-int/get_tools/master/get_tools.py /XMOS/
RUN cd /XMOS \
    && chmod 755 get_tools.py \
    && echo "export PATH=$PATH:/XMOS" \
    >> /etc/profile.d/xmos_tools.sh \
    && chmod a+x /etc/profile.d/xmos_tools.sh

# set login shell
SHELL ["/bin/bash", "-l", "-c"]
