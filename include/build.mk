define PROJECT_HELP_MSG
Usage:
    make help                   show this message
    make build                  make Horovod TF image with Open MPI
    make build-intel            make Horovod TF image with Intel MPI
    make run-mpi				run training using Open MPI image
    make run-mpi-intel			run training using Intel MPI image
    make run					run training in non-distributed mode
    make push					push Horovod TF image with Open MPI
    make push-intel				push Horovod TF image with Intel MPI
endef
export PROJECT_HELP_MSG

DATA_DIR:=/mnt/imagenet
#DATA_DIR:=/mnt/rmdsk
PWD:=$(shell pwd)
FAKE:='False'
FAKE_DATA_LENGTH:=1281167
ROOT:=$(shell dirname ${PWD})


setup_volumes:=-v $(PWD)/src:/mnt/script \
	-v $(DATA_DIR):/mnt/input \
	-v $(DATA_DIR)/temp/model:/mnt/model \
	-v $(DATA_DIR)/temp/output:/mnt/output \
	-v $(ROOT)/common:/mnt/common


setup_environment:=--env AZ_BATCHAI_INPUT_TRAIN='/mnt/input' \
	--env AZ_BATCHAI_INPUT_TEST='/mnt/input' \
	--env AZ_BATCHAI_OUTPUT_MODEL='/mnt/model' \
	--env AZ_BATCHAI_JOB_TEMP_DIR='/mnt/output' \
	--env AZ_BATCHAI_INPUT_SCRIPTS='/mnt/script' \
	--env PYTHONPATH=/mnt/common/:$$PYTHONPATH


define execute_mpi
 nvidia-docker run -it \
 --shm-size="8g" \
 $(setup_volumes) \
 $(setup_environment) \
 --env DISTRIBUTED='True' \
 --env FAKE=$(FAKE) \
 --env FAKE_DATA_LENGTH=$(FAKE_DATA_LENGTH) \
 --privileged \
 $(1) bash -c "mpirun -np 2 -H localhost:2 python $(2)"
endef

define execute_mpi_intel
 nvidia-docker run -it \
 --shm-size="8g" \
 $(setup_volumes) \
 $(setup_environment) \
 --env DISTRIBUTED='True' \
 --env FAKE=$(FAKE) \
 --env FAKE_DATA_LENGTH=$(FAKE_DATA_LENGTH) \
 --privileged \
 $(1) bash -c " source /opt/intel/compilers_and_libraries_2017.4.196/linux/mpi/intel64/bin/mpivars.sh; mpirun -n 2 -host localhost -ppn 2 -env I_MPI_DAPL_PROVIDER=ofa-v2-ib0 -env I_MPI_DYNAMIC_CONNECTION=0 python $(2)"
endef

define execute
 nvidia-docker run -it \
 --shm-size="8g" \
 $(setup_volumes) \
 $(setup_environment) \
 --env DISTRIBUTED='False' \
 --env FAKE=$(FAKE) \
 --env FAKE_DATA_LENGTH=$(FAKE_DATA_LENGTH) \
 $(1) bash -c "python $(2)"
endef


help:
	echo "$$PROJECT_HELP_MSG" | less

build:
	docker build -t $(image) $(dockerfile)

run:
	$(call execute, $(image), $(script))

push:
	docker push $(image)


.PHONY: help build push
