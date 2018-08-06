
DATA_DIR:=/mnt/imagenet
PWD:=$(shell pwd)
FAKE:='False'
FAKE_DATA_LENGTH:=1281167
name_prefix:=caia/distributed-training
image:=$(name_prefix).horovod-tf
dockerfile:=$(PWD)/Docker
script:=\$$AZ_BATCHAI_INPUT_SCRIPTS/imagenet_estimator_tf_horovod.py
include ../include/build.mk

run-mpi:
	$(call execute_mpi_intel, $(image), $(script))