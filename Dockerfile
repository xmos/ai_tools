FROM continuumio/miniconda3:latest

ENV TOOLS_VERSION=15.0.0
ENV TOOLS_PRERELEASE=rc4

ADD environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -afy
ENV PATH="/opt/conda/envs/bin:$PATH"

RUN wget http://intranet/projects/tools/ReleasesTools/${TOOLS_VERSION}_${TOOLS_PRERELEASE}/Linux64_xTIMEcomposer_${TOOLS_VERSION}.tgz
RUN cd / && tar xvf Linux64_xTIMEcomposer_${TOOLS_VERSION}.tgz
RUN echo "pushd /XMOS/xTIMEcomposer/${TOOLS_VERSION} && . SetEnv && popd" >> /etc/profile.d/xmos_tools.sh \
    && chmod a+x /etc/profile.d/xmos_tools.sh

CMD /bin/bash
