define PROJECT_HELP_MSG
Usage:
    make help                   show this message
    make build                  build docker image
    make push					 push container
    make run					 run benchmarking container
endef
export PROJECT_HELP_MSG
PWD:=$(shell pwd)
dockerhub:=
image_name:=$(dockerhub)/distributed-training-control

help:
	echo "$$PROJECT_HELP_MSG" | less

build:
	docker build -t $(image_name) Docker

run:
	docker run -p 9999:9999 -v $(PWD):/workspace -it $(image_name) bash

push:
	docker push $(image_name)



.PHONY: help build push
