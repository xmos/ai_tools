FROM --platform=linux/amd64 jupyter/base-notebook:2023-01-30

# switch to and install opencv dependencies
USER root
RUN sudo apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
USER ${NB_USER}

RUN pip install --upgrade pip

# install all python dependencies
COPY --chown=${NB_UID}:${NB_GID} ./requirements.txt /tmp/
RUN pip install --no-cache-dir --requirement /tmp/requirements.txt && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"
RUN pip install --no-cache-dir nvidia-pyindex  && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"
RUN pip install --no-cache-dir onnx-graphsurgeon  && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# copy kaggle api credentials
# install all python dependencies
COPY --chown=${NB_UID}:${NB_GID} --chmod=600 ./kaggle.json /home/${NB_USER}/.kaggle/kaggle.json

ENV JUPYTER_ENABLE_LAB=yes
