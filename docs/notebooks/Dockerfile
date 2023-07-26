FROM jupyter/minimal-notebook

RUN pip install --upgrade pip

# COPY ./requirements.txt ./requirements.txt 
# RUN pip install nvidia-pyindex
# RUN pip install onnx-graphsurgeon
# RUN pip install -r ./requirements.txt 

COPY --chown=${NB_UID}:${NB_GID} ./requirements.txt /tmp/
RUN pip install --quiet --no-cache-dir --requirement /tmp/requirements.txt && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

ENV JUPYTER_ENABLE_LAB=yes