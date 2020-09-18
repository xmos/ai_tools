FROM continuumio/miniconda3:latest

# fix conda perms
RUN chmod -R 777 /opt/conda \
    && mkdir -p /.conda \
    && chmod -R 777 /.conda

# install get_tools script and tools lib dependencies
RUN apt-get update && apt-get install -y libncurses5 libncurses5-dev environment-modules && apt-get clean autoclean
RUN mkdir -m 777 /XMOS && cd /XMOS \
    && wget -q https://github0.xmos.com/raw/xmos-int/get_tools/master/get_tools.py \
    && chmod a+x get_tools.py
RUN echo "export MODULEPATH=/XMOS/modulefiles:/XMOS/template_modulefiles" \
    >> /etc/profile.d/xmos_tools.sh \
    && chmod a+x /etc/profile.d/xmos_tools.sh

# update conda install
RUN conda update -n base -c defaults conda

# set login shell
SHELL ["/bin/bash", "-l", "-c"]
