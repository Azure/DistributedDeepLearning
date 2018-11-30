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

# # Create Docker Image for PyTorch
# In this notebook we will create the image for our PyTorch script to run in. We will go through the process of creating the image and testing it locally to make sure it runs before submitting it to the cluster. It is often recommended you do this rather than debugging on the cluster since debugging on a cluster can be much more difficult and time consuming.

# +
import sys
sys.path.append("../common") 

from dotenv import get_key
import os
from utils import dotenv_for
import docker
# -

# Below are the variables that describe our experiment. By default we are using the NC24rs_v3 (Standard_NC24rs_v3) VMs which have V100 GPUs and Infiniband. By default we are using 2 nodes with each node having 4 GPUs, this equates to 8 GPUs. Feel free to increase the number of nodes but be aware what limitations your subscription may have.
#
# Set the USE_FAKE to True if you want to use fake data rather than the Imagenet dataset. This is often a good way to debug your models as well as checking what IO overhead is.

# + {"tags": ["parameters"]}
dotenv_path = dotenv_for()
USE_FAKE               = True
DOCKERHUB              = os.getenv('DOCKER_REPOSITORY', "masalvar")  #"<YOUR DOCKERHUB>"
NUM_PROCESSES          = 2
DOCKER_PWD             = get_key(dotenv_path, 'DOCKER_PWD')
# -

dc = docker.from_env()

image, log_iter = dc.images.build(path='Docker', 
                          tag='{}/caia-horovod-pytorch'.format(DOCKERHUB))

image.tags[0]

# +
container_labels = {'containerName': 'pytorchgpu'}
environment ={
    "DISTRIBUTED":True,
    "PYTHONPATH":'/workspace/common/',
}

volumes = {
    os.getenv('EXT_PWD'): {
                                'bind': '/workspace', 
                                'mode': 'rw'
                               }
}

if USE_FAKE:
    environment['FAKE'] = True
else:
    environment['FAKE'] = False
    volumes[os.getenv('EXT_DATA')]={'bind': '/mnt/input', 'mode': 'rw'}
    environment['AZ_BATCHAI_INPUT_TRAIN'] = '/mnt/input/train'
    environment['AZ_BATCHAI_INPUT_TEST'] = '/mnt/input/validation'
# -

cmd=f'mpirun -np {NUM_PROCESSES} -H localhost:{NUM_PROCESSES} '\
     'python -u /workspace/HorovodPytorch/src/imagenet_pytorch_horovod.py'
container = dc.containers.run(image.tags[0], 
                              command=cmd,
                              detach=True, 
                              labels=container_labels,
                              runtime='nvidia',
                              volumes=volumes,
                              environment=environment,
                              shm_size='8G',
                              privileged=True)

for line in container.logs(stderr=True, stream=True):
    print(line.decode("utf-8"),end ="")

container.reload() # Refresh state
if container.status is 'running':
    container.kill()

for line in dc.images.push(image.tags[0], 
                           stream=True,
                           auth_config={"username": DOCKERHUB,
                                        "password": DOCKER_PWD}):
    print(line)
