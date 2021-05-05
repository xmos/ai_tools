FROM continuumio/miniconda3:4.8.2

# This Dockerfile is for use by the XMOS CI system
# It provides a minimal environment needed to execute the Jenkinsfile
# Most of the dependecies here are handled conda so we only include:
#  - conda setup
#  - xmos tools setup

# fix conda perms and config
RUN chmod -R 777 /opt/conda \
    && mkdir -p /.conda \
    && chmod -R 777 /.conda \
    && conda init \
    && conda config --set auto_activate_base false

# install tools lib dependencies
RUN apt-get update && apt-get install -y \
    libncurses5 libncurses5-dev \
    tcl environment-modules \
    && apt-get clean autoclean

# install clang
RUN apt-get update && apt-get install -y \
    gnupg lsb-release software-properties-common
ADD https://apt.llvm.org/llvm.sh /tmp/
ARG clang_version=12
RUN cd /tmp \
    && chmod +x llvm.sh \
    && ./llvm.sh $clang_version
RUN ln -s /usr/bin/clang-$clang_version /usr/bin/clang \
    && ln -s /usr/bin/clang++-$clang_version /usr/bin/clang++ \
    && ln -s /usr/bin/clang /usr/bin/cc \
    && ln -s /usr/bin/clang++ /usr/bin/c++

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
