FROM continuumio/miniconda3:latest

ENV TOOLS_VERSION=15.0.0rc4
ENV TOOLS_SHORT_VERSION=15.0.0

ADD environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -afy

# Pull the environment name out of the environment.`yml
RUN echo "source activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH="/opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH"


RUN wget http://intranet/projects/tools/ReleasesTools/${TOOLS_VERSION}/Linux64_xTIMEcomposer_${TOOLS_SHORT_VERSION}.tgz
RUN tar xvf Linux64_xTIMEcomposer_${TOOLS_SHORT_VERSION}.tgz /
RUN echo "pushd /XMOS/xTIMEcomposer/${TOOLS_SHORT_VERSION} && . SetupEnv && popd" > ~/.bashrc

CMD /bin/bash
