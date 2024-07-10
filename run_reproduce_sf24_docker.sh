#!/bin/bash
#USER_ID=$(id -u $USER)
if docker info >/dev/null 2>&1; then
	DIR=$(readlink -f shared)
	if [ -z "$(docker images -q sf24/malsyngen:latest 2> /dev/null)" ]; then
	docker build -t sf24/malsyngen:latest . 
	fi
	docker run -it --name=MalSynGen-$RANDOM -v $DIR:/SynTabData/shared -e DISPLAY=unix$DISPLAY sf24/malsyngen:latest /MalSynGen/run_reproduce_sf24.sh
	#ls shared/outputs/
else
    echo "Seu usuário atual não possui permissões para executar docker sem sudo, execute o seguinte comando e reinicialize a máquina: sudo usermod -aG docker SEU_USER "
fi
