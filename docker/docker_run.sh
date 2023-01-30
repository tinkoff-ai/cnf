#!/usr/bin/env bash
docker pull pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
container_name=flows

nvidia-docker stop ${container_name}
nvidia-docker rm ${container_name}
nvidia-docker run -it -d --net=host --ipc=host \
-v $PWD/../:/flows \
-w /flows --name ${container_name} pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime bash
