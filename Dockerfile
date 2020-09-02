FROM continuumio/miniconda3:latest

# ARG CONDA_ENV=.venv
ENV TOOLS_VERSION=15.0.0
ENV TOOLS_PRERELEASE=rc4

RUN chmod -R 777 /opt/conda

RUN apt-get update && apt-get install -y libncurses5 libncurses5-dev && apt-get clean autoclean
RUN wget -q http://intranet/projects/tools/ReleasesTools/${TOOLS_VERSION}_${TOOLS_PRERELEASE}/Linux64_xTIMEcomposer_${TOOLS_VERSION}.tgz \
    && cd / \
    && tar xf Linux64_xTIMEcomposer_${TOOLS_VERSION}.tgz \
    && chmod -R 777 /XMOS \
    && rm -f Linux64_xTIMEcomposer_${TOOLS_VERSION}.tgz
RUN echo "pushd /XMOS/xTIMEcomposer/${TOOLS_VERSION} > /dev/null && . SetEnv && popd > /dev/null" \
    >> /etc/profile.d/xmos_tools.sh \
    && chmod a+x /etc/profile.d/xmos_tools.sh

run echo "/bin/bash -l -c $@" > /enter.sh \
    && chmod a+x /enter.sh

# ADD environment.yml /tmp/environment.yml
# RUN conda env create -n $CONDA_ENV -f /tmp/environment.yml && conda clean -afy
# RUN echo "source activate $CONDA_ENV" >> /etc/bash.bashrc
# ENV PATH /opt/conda/envs/$CONDA_ENV/bin:$PATH
# ENV CONDA_DEFAULT_ENV $CONDA_ENV

ENTRYPOINT ["/enter.sh"]
