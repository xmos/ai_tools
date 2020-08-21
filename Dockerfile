FROM continuumio/miniconda3:latest

ADD environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -afy
RUN chmod -R 777 /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)

# Pull the environment name out of the environment.yml
RUN echo "source activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" > ~/.bashrc

CMD /bin/bash
