define PROJECT_HELP_MSG
Usage:
    make help                   show this message
    make build                  make Horovod TF image with Open MPI
    make push					push Horovod TF image with Open MPI
endef
export PROJECT_HELP_MSG

help:
	echo "$$PROJECT_HELP_MSG" | less

build:
	docker build -t $(image) $(dockerpath)

push:
	docker push $(image)


.PHONY: help build push
