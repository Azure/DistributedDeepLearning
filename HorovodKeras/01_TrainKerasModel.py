# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Train Keras Model Distributed on Batch AI
# In this notebook we will train a Keras model ([ResNet50](https://arxiv.org/abs/1512.03385)) in a distributed fashion using [Horovod](https://github.com/uber/horovod) on the Imagenet dataset. This tutorial will take you through the following steps:
#  * [Create Experiment](#experiment)
#  * [Upload Training Scripts](#training_scripts)
#  * [Submit and Monitor Job](#job)
#  * [Clean Up Resources](#clean_up)

# +
import sys
sys.path.append("../common") 

import json
from dotenv import get_key
import os
from utils import write_json_to_file, dotenv_for
# -

# Set the USE_FAKE to True if you want to use fake data rather than the ImageNet dataset. This is often a good way to debug your models as well as checking what IO overhead is.

# + {"tags": ["parameters"]}
# Variables for Batch AI - change as necessary
dotenv_path = dotenv_for()
GROUP_NAME             = get_key(dotenv_path, 'GROUP_NAME')
FILE_SHARE_NAME        = get_key(dotenv_path, 'FILE_SHARE_NAME')
WORKSPACE              = get_key(dotenv_path, 'WORKSPACE')
NUM_NODES              = int(get_key(dotenv_path, 'NUM_NODES'))
CLUSTER_NAME           = get_key(dotenv_path, 'CLUSTER_NAME')
GPU_TYPE               = get_key(dotenv_path, 'GPU_TYPE')
PROCESSES_PER_NODE     = int(get_key(dotenv_path, 'PROCESSES_PER_NODE'))
STORAGE_ACCOUNT_NAME   = get_key(dotenv_path, 'STORAGE_ACCOUNT_NAME')

EXPERIMENT             = f"distributed_keras_{GPU_TYPE}"
USE_FAKE               = False
DOCKERHUB              = os.getenv('DOCKER_REPOSITORY', "masalvar")
# -

FAKE='-x FAKE=True' if USE_FAKE else ''
TOTAL_PROCESSES = PROCESSES_PER_NODE * NUM_NODES

# <a id='experiment'></a>
# # Create Experiment
# Next we create our experiment.

!az batchai experiment create -n $EXPERIMENT -g $GROUP_NAME -w $WORKSPACE

# <a id='training_scripts'></a>
# # Upload Training Scripts
# We need to upload our training scripts and associated files

json_data = !az storage account keys list -n $STORAGE_ACCOUNT_NAME -g $GROUP_NAME
storage_account_key = json.loads(''.join([i for i in json_data if 'WARNING' not in i]))[0]['value']

# + {"tags": ["stripout"]}
# %env AZURE_STORAGE_ACCOUNT $STORAGE_ACCOUNT_NAME
# %env AZURE_STORAGE_KEY=$storage_account_key
# -

# Upload our training scripts

!az storage file upload --share-name $FILE_SHARE_NAME --source src/imagenet_keras_horovod.py --path scripts
!az storage file upload --share-name $FILE_SHARE_NAME --source src/data_generator.py --path scripts
!az storage file upload --share-name $FILE_SHARE_NAME --source ../common/timer.py --path scripts

# Let's check our cluster we created earlier

!az batchai cluster list -w $WORKSPACE -o table

# <a id='job'></a>
# ## Submit and Monitor Job
# Below we specify the job we wish to execute.  

jobs_dict = {
  "$schema": "https://raw.githubusercontent.com/Azure/BatchAI/master/schemas/2017-09-01-preview/job.json",
  "properties": {
    "nodeCount": NUM_NODES,
    "customToolkitSettings": {
      "commandLine": f"echo $AZ_BATCH_HOST_LIST; \
    cat $AZ_BATCHAI_MPI_HOST_FILE; \
    mpirun -np {TOTAL_PROCESSES} --hostfile $AZ_BATCHAI_MPI_HOST_FILE \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH \
    -mca btl_tcp_if_include eth0 \
    -x NCCL_SOCKET_IFNAME=eth0 \
    -mca btl ^openib \
    -x NCCL_IB_DISABLE=1 \
    -x DISTRIBUTED=True \
    -x AZ_BATCHAI_INPUT_TRAIN \
    -x AZ_BATCHAI_INPUT_TEST \
    --allow-run-as-root \
      {FAKE} \
      python -u $AZ_BATCHAI_INPUT_SCRIPTS/imagenet_leras_horovod.py"
    },
    "stdOutErrPathPrefix": "$AZ_BATCHAI_MOUNT_ROOT/extfs",
    "inputDirectories": [{
        "id": "SCRIPTS",
        "path": "$AZ_BATCHAI_MOUNT_ROOT/extfs/scripts"
      },
      {
        "id": "TRAIN",
        "path": "$AZ_BATCHAI_MOUNT_ROOT/nfs/imagenet",
      },
      {
        "id": "TEST",
        "path": "$AZ_BATCHAI_MOUNT_ROOT/nfs/imagenet",
      },
    ],
    "outputDirectories": [{
        "id": "MODEL",
        "pathPrefix": "$AZ_BATCHAI_MOUNT_ROOT/extfs",
        "pathSuffix": "Models"
    }],
    "containerSettings": {
      "imageSourceRegistry": {
        "image": f"{DOCKERHUB}/caia-horovod-keras"
      }
    }
  }
}

write_json_to_file(jobs_dict, 'job.json')

JOB_NAME='keras-horovod-{}'.format(NUM_NODES*PROCESSES_PER_NODE)

# We now submit the job to Batch AI

# + {"tags": ["stripout"]}
!az batchai job create -n $JOB_NAME --cluster $CLUSTER_NAME -w $WORKSPACE -e $EXPERIMENT -f job.json
# -

# With the command below we can check the status of the job

!az batchai job list -w $WORKSPACE -e $EXPERIMENT -o table

# To view the files that the job has generated use the command below

# + {"tags": ["stripout"]}
!az batchai job file list -w $WORKSPACE -e $EXPERIMENT --j $JOB_NAME --output-directory-id stdouterr
# -

# We are also able to stream the stdout and stderr that our job produces. This is great to check the progress of our job as well as debug issues.

# + {"tags": ["stripout"]}
!az batchai job file stream -w $WORKSPACE -e $EXPERIMENT --j $JOB_NAME --output-directory-id stdouterr -f stdout.txt

# + {"tags": ["stripout"]}
!az batchai job file stream -w $WORKSPACE -e $EXPERIMENT --j $JOB_NAME --output-directory-id stdouterr -f stderr.txt
# -

# We can either wait for the job to complete or delete it with the command below.

!az batchai job delete -w $WORKSPACE -e $EXPERIMENT --name $JOB_NAME -y

# <a id='clean_up'></a>
# ## Clean Up Resources
# Next we wish to tidy up the resource we created.  
# First we reset the default values we set earlier.

!az configure --defaults group=''
!az configure --defaults location=''

#  Next we delete the cluster

!az batchai cluster delete -w $WORKSPACE --name $CLUSTER_NAME -g $GROUP_NAME -y

# Once the cluster is deleted you will not incur any cost for the computation but you can still retain your experiments and workspace. If you wish to delete those as well execute the commands below.

!az batchai experiment delete -w $WORKSPACE --name $EXPERIMENT -g $GROUP_NAME -y

!az batchai workspace delete -n $WORKSPACE -g $GROUP_NAME -y

# Finally we can delete the group and we will have deleted everything created for this tutorial.

!az group delete --name $GROUP_NAME -y
