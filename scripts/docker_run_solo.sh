#!/bin/bash
if [ -z "$(sudo docker images -q sf24/malsyngen:latest 2> /dev/null)" ]; then
 ./scripts/docker_build.sh
fi

sudo docker run -it --name=MalSynGen-$RANDOM -e DISPLAY=unix$DISPLAY sf24/malsyngen:latest
