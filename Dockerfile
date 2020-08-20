FROM continuumio/miniconda3:latest

RUN apt-get update && apt-get install -y \
    cmake

COPY environment.yml .
RUN conda env create --name xmos -f environment.yml python=3.8

CMD /bin/bash
