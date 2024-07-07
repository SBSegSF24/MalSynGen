#!/bin/bash
[ $1 ] && [ -d $1 ] || {
	echo "Usage: $0 <code_directory>"
	echo " example: $0 ."
	exit
}
sudo docker run -it --name=MalSynGen-$RANDOM -v $(readlink -f $1):/MalSynGen/shared -e DISPLAY=unix$DISPLAY sf24/malsyngen:latest bash 
