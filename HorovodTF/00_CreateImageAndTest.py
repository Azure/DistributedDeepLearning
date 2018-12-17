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

# # Create Docker Image for TensorFlow
# In this notebook we will create the Docker image for our TensorFlow script to run in. We will go through the process of creating the image and testing it locally to make sure it runs before submitting it to the cluster. It is often recommended you do this rather than debugging on the cluster since debugging on a cluster can be much more difficult and time consuming.
#  
# **You will need to be running everything on a GPU enabled VM to run this notebook.** 

# +
import sys
sys.path.append("../common") 

from dotenv import get_key
import os
from utils import dotenv_for
import docker
# -

# We will use fake data here since we don't want to have to download the data etc. Using fake data is often a good way to debug your models as well as checking what IO overhead is. Here we are setting the number of processes (NUM_PROCESSES) to 2 since the VM we are testing on has 2 GPUs. If you are running on a machine with 1 GPU set NUM_PROCESSES to 1.

# + {"tags": ["parameters"]}
dotenv_path = dotenv_for()
USE_FAKE               = True
DOCKERHUB              = os.getenv('DOCKER_REPOSITORY', "masalvar")
NUM_PROCESSES          = 2
DOCKER_PWD             = get_key(dotenv_path, 'DOCKER_PWD')
# -

dc = docker.from_env()

image, log_iter = dc.images.build(path='Docker', 
                          tag='{}/caia-horovod-tensorflow'.format(DOCKERHUB))

# +
container_labels = {'containerName': 'tensorflowgpu'}
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
     'python -u /workspace/HorovodTF/src/imagenet_estimator_tf_horovod.py'
container = dc.containers.run(image.tags[0], 
                              command=cmd,
                              detach=True, 
                              labels=container_labels,
                              runtime='nvidia',
                              volumes=volumes,
                              environment=environment,
                              shm_size='8G',
                              privileged=True)

# With the code below we are simply monitoring what is happening in the container. Feel free to stop the notebook when you are happy that everything is working.

# + {"tags": ["stripout"]}
for line in container.logs(stderr=True, stream=True):
    print(line.decode("utf-8"),end ="")
# -

container.reload() # Refresh state
if container.status is 'running':
    container.kill()

# + {"tags": ["stripout"]}
for line in dc.images.push(image.tags[0], 
                           stream=True,
                           auth_config={"username": DOCKERHUB,
                                        "password": DOCKER_PWD}):
    print(line)
