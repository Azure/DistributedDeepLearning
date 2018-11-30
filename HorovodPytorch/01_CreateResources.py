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

# # Create Azure and Batch AI Resources
# In this notebook we will create the necessary resources to train a ResNet50 model([ResNet50](https://arxiv.org/abs/1512.03385)) in a distributed fashion using [Horovod](https://github.com/uber/horovod) on the Imagenet dataset. If you plan on using fake data then the sections marked optional can be skipped. This notebook will take you through the following steps:
#  * [Create Azure Resources](#azure_resources)
#  * [Create Fileserver(NFS)(Optional)](#create_fileshare)
#  * [Upload Data to Blob (Optional)](#upload_data)
#  * [Configure Batch AI Cluster](#configure_cluster)

# +
import sys
sys.path.append("../common") 

from dotenv import dotenv_values, set_key, find_dotenv, get_key
from getpass import getpass
import os
import json
from utils import get_password, write_json_to_file, dotenv_for
from pathlib import Path
# -

# Below are the variables that describe our experiment. By default we are using the NC24rs_v3 (Standard_NC24rs_v3) VMs which have V100 GPUs and Infiniband. By default we are using 2 nodes with each node having 4 GPUs, this equates to 8 GPUs. Feel free to increase the number of nodes but be aware what limitations your subscription may have.
#
# Set the USE_FAKE to True if you want to use fake data rather than the Imagenet dataset. This is often a good way to debug your models as well as checking what IO overhead is.

# + {"tags": ["parameters"]}
# Variables for Batch AI - change as necessary
ID                     = "ddpytorch"
GROUP_NAME             = f"batch{ID}rg"
STORAGE_ACCOUNT_NAME   = f"batch{ID}st"
FILE_SHARE_NAME        = f"batch{ID}share"
SELECTED_SUBSCRIPTION  = "Boston Team Danielle" #"<YOUR SUBSCRIPTION>"
WORKSPACE              = "workspace"
NUM_NODES              = 2
CLUSTER_NAME           = "msv100"
VM_SIZE                = "Standard_NC24rs_v3"
GPU_TYPE               = "V100"
PROCESSES_PER_NODE     = 4
LOCATION               = "eastus"
NFS_NAME               = f"batch{ID}nfs"
EXPERIMENT             = f"distributed_pytorch_{GPU_TYPE}"
USERNAME               = "batchai_user"
USE_FAKE               = False
DOCKERHUB              = "masalvar" #"<YOUR DOCKERHUB>"
DATA                   = Path("/data/imagenet")
CONTAINER_NAME         = f"batch{ID}container"
# -

FAKE='-env FAKE=True' if USE_FAKE else ''
TOTAL_PROCESSES = PROCESSES_PER_NODE * NUM_NODES

# <a id='azure_resources'></a>
# ## Create Azure Resources
# First we need to log in to our Azure account. 

!az login -o table

# If you have more than one Azure account you will need to select it with the command below. If you only have one account you can skip this step.

!az account set --subscription "$SELECTED_SUBSCRIPTION"

!az account list -o table

# Next we create the group that will hold all our Azure resources.

!az group create -n $GROUP_NAME -l $LOCATION -o table

# We will create the storage account that will store our fileshare where all the outputs from the jobs will be stored.

json_data = !az storage account create -l $LOCATION -n $STORAGE_ACCOUNT_NAME -g $GROUP_NAME --sku Standard_LRS
print('Storage account {} provisioning state: {}'.format(STORAGE_ACCOUNT_NAME, 
                                                         json.loads(''.join(json_data))['provisioningState']))

json_data = !az storage account keys list -n $STORAGE_ACCOUNT_NAME -g $GROUP_NAME
storage_account_key = json.loads(''.join([i for i in json_data if 'WARNING' not in i]))[0]['value']

!az storage share create --account-name $STORAGE_ACCOUNT_NAME \
--account-key $storage_account_key --name $FILE_SHARE_NAME

!az storage directory create --share-name $FILE_SHARE_NAME  --name scripts \
--account-name $STORAGE_ACCOUNT_NAME --account-key $storage_account_key

# Here we are setting some defaults so we don't have to keep adding them to every command

!az configure --defaults location=$LOCATION
!az configure --defaults group=$GROUP_NAME

# %env AZURE_STORAGE_ACCOUNT $STORAGE_ACCOUNT_NAME
# %env AZURE_STORAGE_KEY=$storage_account_key

# #### Create Workspace
# Batch AI has the concept of workspaces and experiments. Below we will create the workspace for our work.

!az batchai workspace create -n $WORKSPACE -g $GROUP_NAME

# <a id='upload_data'></a>
# ## Upload Data to Blob (Optional)
# In this section we will create a blob container and upload the imagenet data we prepared locally in the previous notebook.

!az storage container create --account-name {STORAGE_ACCOUNT_NAME} \
                             --account-key {storage_account_key} \
                             --name {CONTAINER_NAME}

# Should take about 20 minnutes
!azcopy --source {DATA/"train.tar.gz"} \
--destination https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{CONTAINER_NAME}/train.tar.gz \
--dest-key {storage_account_key} --quiet

!azcopy --source {DATA/"validation.tar.gz"} \
--destination https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{CONTAINER_NAME}/validation.tar.gz \
--dest-key {storage_account_key} --quiet

# <a id='create_fileshare'></a>
# ## Create Fileserver (Optional)
# In this example we will store the data on an NFS fileshare. It is possible to use many storage solutions with Batch AI. NFS offers the best traideoff between performance and ease of use. The best performance is achieved by loading the data locally but this can be cumbersome since it requires that the data is download by the all the nodes which with the imagenet dataset can take hours. 

!az batchai file-server create -n $NFS_NAME --disk-count 4 --disk-size 250 -w $WORKSPACE \
-s Standard_DS4_v2 -u $USERNAME -p {get_password(dotenv_for())} -g $GROUP_NAME --storage-sku Premium_LRS

!az batchai file-server list -o table -w $WORKSPACE -g $GROUP_NAME

json_data = !az batchai file-server list -w $WORKSPACE -g $GROUP_NAME
nfs_ip=json.loads(''.join([i for i in json_data if 'WARNING' not in i]))[0]['mountSettings']['fileServerPublicIp']

# After we have created the NFS share we need to copy the data to it. To do this we write the script below which will be executed on the fileserver. It installs a tool called azcopy and then downloads and extracts the data to the appropriate directory.

nodeprep_script = f"""
#!/usr/bin/env bash
wget https://gist.githubusercontent.com/msalvaris/073c28a9993d58498957294d20d74202/raw/87a78275879f7c9bb8d6fb9de8a2d2996bb66c24/install_azcopy
chmod 777 install_azcopy
sudo ./install_azcopy

mkdir -p /data/imagenet

azcopy --source https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{CONTAINER_NAME}/validation.tar.gz \
        --destination  /data/imagenet/validation.tar.gz\
        --source-key {storage_account_key}\
        --quiet


azcopy --source https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{CONTAINER_NAME}/train.tar.gz \
        --destination  /data/imagenet/train.tar.gz\
        --source-key {storage_account_key}\
        --quiet

cd /data/imagenet
tar -xzf train.tar.gz
tar -xzf validation.tar.gz
"""

with open('nodeprep.sh', 'w') as f:
    f.write(nodeprep_script)

# Next we will copy the file over and run it on the NFS VM. This will install azcopy and download and prepare the data

!sshpass -p {get_password(dotenv_for())} scp -o "StrictHostKeyChecking=no" nodeprep.sh $USERNAME@{nfs_ip}:~/

!sshpass -p {get_password(dotenv_for())} ssh -o "StrictHostKeyChecking=no" $USERNAME@{nfs_ip} "sudo chmod 777 ~/nodeprep.sh && ./nodeprep.sh"

# <a id='configure_cluster'></a>
# ## Configure Batch AI Cluster
# We then upload the scripts we wish to execute onto the fileshare. The fileshare will later be mounted by Batch AI. An alternative to uploading the scripts would be to embedd them inside the Docker container.

# Below it the command to create the cluster.

!az batchai cluster create \
    -w $WORKSPACE \
    --name $CLUSTER_NAME \
    --image UbuntuLTS \
    --vm-size $VM_SIZE \
    --min $NUM_NODES --max $NUM_NODES \
    --afs-name $FILE_SHARE_NAME \
    --afs-mount-path extfs \
    --user-name $USERNAME \
    --password {get_password(dotenv_for())} \
    --storage-account-name $STORAGE_ACCOUNT_NAME \
    --storage-account-key $storage_account_key \
    --nfs $NFS_NAME \
    --nfs-mount-path nfs \
    --config-file cluster_config/cluster.json

# Let's check that the cluster was created succesfully.

!az batchai cluster show -n $CLUSTER_NAME -w $WORKSPACE

!az batchai cluster list -w $WORKSPACE -o table

!az batchai cluster node list -c $CLUSTER_NAME -w $WORKSPACE -o table
