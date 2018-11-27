define PROJECT_HELP_MSG
Usage:
    make help                   show this message
    make build                  build docker image
    make push					 push container
    make run					 run benchmarking container
    make jupyter                run jupyter notebook inside container
endef
export PROJECT_HELP_MSG
PWD:=$(shell pwd)
dockerhub:=
image_name:=$(dockerhub)/distributed-training-control

help:
	echo "$$PROJECT_HELP_MSG" | less

build:
	docker build -t $(image_name) Docker

jupyter:
	docker run -p 9999:9999 \
	           -v $(PWD):/workspace \
	           -v /var/run/docker.sock:/var/run/docker.sock \
	           -it $(image_name) bash -c "jupyter notebook"

run:
	docker run -p 9999:9999 \
	           -v $(PWD):/workspace \
	           -v /var/run/docker.sock:/var/run/docker.sock \
	           -it $(image_name)

push:
	docker push $(image_name)



.PHONY: help build push
