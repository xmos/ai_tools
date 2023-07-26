FROM --platform=linux/amd64 jupyter/base-notebook:2023-01-30

RUN pip install --upgrade pip

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


ENV JUPYTER_ENABLE_LAB=yes
