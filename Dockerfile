FROM continuumio/miniconda3:latest

ARG USER=xmos
ARG UID=1000

RUN apt-get update && apt-get install -y cmake
RUN adduser --disabled-password --gecos "" --uid $UID $USER

RUN mkdir /opt/conda/envs/xmos /opt/conda/pkgs && \
    chgrp xmos /opt/conda/pkgs && \
    chmod g+w /opt/conda/pkgs && \
    touch /opt/conda/pkgs/urls.txt && \
    chown xmos /opt/conda/envs/xmos /opt/conda/pkgs/urls.txt


WORKDIR /home/xmos
USER xmos

COPY --chown $USER:$USER environment.yml .
RUN conda env create --name xmos -f environment.yml python=3.8

CMD /bin/bash
