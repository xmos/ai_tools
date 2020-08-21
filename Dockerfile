FROM continuumio/miniconda3:latest

RUN apt-get update && apt-get install -y \
    cmake

ADD environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml -p /opt/conda/envs && conda clean -afy

RUN conda init bash

ENV PATH /opt/conda/envs/bin:$PATH

CMD /bin/bash
