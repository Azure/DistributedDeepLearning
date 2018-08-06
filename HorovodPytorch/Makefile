DATA_DIR:=/mnt/imagenet
PWD:=$(shell pwd)
FAKE:='False'
FAKE_DATA_LENGTH:=1281167
name_prefix:=caia/distributed-training
image:=$(name_prefix).horovod-pytorch
dockerfile:=$(PWD)/Docker
script:=\$$AZ_BATCHAI_INPUT_SCRIPTS/imagenet_pytorch_horovod.py
include ../include/build.mk

run-mpi:
	$(call execute_mpi, $(image), $(script))
