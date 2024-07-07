#!/bin/bash
sudo docker run -it --name=MalSynGen-$RANDOM -e DISPLAY=unix$DISPLAY sf24/malsyngen:latest
