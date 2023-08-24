#!/bin/sh
docker build . --tag ai_tools_docs
docker run                      `# Create a container from an image and runs a command in it.` \
  --rm \
  -v $(pwd):/home/jovyan/work:z `# Bind the current directory to the workdir in the container. The 'z' causes SELinux relabelling of the host directory to allow access by this container (and potentially others).` \
  --name notebooks              `# ` \
  -p 8888:8888                  `# Run on port 8888` \
  -it                          `# Launch in interactive mode` \
  ai_tools_docs                 `# Use image that was just built` \
  start.sh jupyter lab --NotebookApp.token='' --NotebookApp.password='' \