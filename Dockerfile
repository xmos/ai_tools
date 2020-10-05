FROM continuumio/miniconda3:4.8.2

# This Dockerfile is for use by the XMOS CI system
# It provides a minimal environment needed to execute the Jenkinsfile
# Most of the dependecies here are handled conda so we only include:
#  - xmos tools setup
#  - matching user creation
#  - conda permissions

# install tools lib dependencies
RUN apt-get update && apt-get install -y \
    libncurses5 libncurses5-dev \
    tcl environment-modules \
    && apt-get clean autoclean
# install get_tools.py script
#   requires connection to XMOS network at build and run time
#   if not possible, find another way to install the tools
RUN mkdir -m 777 /XMOS && cd /XMOS \
    && wget -q https://github0.xmos.com/raw/xmos-int/get_tools/master/get_tools.py \
    && chmod a+x get_tools.py \
    && echo "export MODULES_SILENT_SHELL_DEBUG=1\nexport MODULEPATH=/XMOS/modulefiles:/XMOS/template_modulefiles\nexport PATH=$PATH:/XMOS" \
    >> /etc/profile.d/xmos_tools.sh \
    && chmod a+x /etc/profile.d/xmos_tools.sh

# Take in ARGS from `docker build --build-arg USER=....`
# Then create group and user so home dirs and perms work
ARG USER=root
ARG UID=0
ARG GID=0
RUN groupadd -g $GID $USER && \
    useradd $USER -u $UID -g $GID -b /home -m

# fix conda perms
RUN chown -R $USER /opt/conda

# Set default user
USER $USER

# set login shell
SHELL ["/bin/bash", "-l", "-c"]
