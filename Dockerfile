FROM continuumio/miniconda3:latest

# ARG CONDA_ENV=.venv
ENV TOOLS_VERSION=15.0.0
ENV TOOLS_PRERELEASE=rc4

RUN apt-get update && apt-get install -y libncurses5 libncurses5-dev && apt-get clean autoclean
RUN wget http://intranet/projects/tools/ReleasesTools/${TOOLS_VERSION}_${TOOLS_PRERELEASE}/Linux64_xTIMEcomposer_${TOOLS_VERSION}.tgz
RUN cd / && tar xvf Linux64_xTIMEcomposer_${TOOLS_VERSION}.tgz
RUN echo "pushd /XMOS/xTIMEcomposer/${TOOLS_VERSION} > /dev/null && . SetEnv && popd > /dev/null" >> /etc/bash.bashrc

# ADD environment.yml /tmp/environment.yml
# RUN conda env create -n $CONDA_ENV -f /tmp/environment.yml && conda clean -afy
# RUN echo "source activate $CONDA_ENV" >> /etc/bash.bashrc
# ENV PATH /opt/conda/envs/$CONDA_ENV/bin:$PATH
# ENV CONDA_DEFAULT_ENV $CONDA_ENV

CMD /bin/bash
