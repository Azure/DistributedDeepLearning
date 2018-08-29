# Training Distributed Training on Batch AI
Object recognition in images is a widely applied technique in computer vision applications. It is often implemented by training a convolutional deep neural network (CNN). The training process can take up to weeks on a single GPU, not to mention the the prohibitively long time needed when performing hyperparameter tuning or experimenting with model architectures.

This repo shows how to train a CNN model in a distributed fashion using [Azure Batch AI](https://docs.microsoft.com/en-us/azure/batch-ai/overview), a managed service that enables deep learning (DL) models to be trained on clusters of Azure virtual machines, including VMs with GPU support. 

We train CNN models ([ResNet50](https://arxiv.org/abs/1512.03385)) using [Horovod](https://github.com/uber/horovod) on the [Imagenet](http://www.image-net.org/) dataset. When training CNN models, we use three DL frameworks for you to choose from: TensorFlow, Keras, or PyTorch.  

To get started with the tutorial, please proceed with following steps **in sequential order**.

 * [Prerequisites](#prerequisites)
 * [Setup](#setup)
 * [TensorFlow version](./HorovodTF)  or [Keras version](./HorovodKeras), or [PyTorch version](./HorovodPytorch) 

<a id='prerequisites'></a>
## Prerequisites
* Local host machine OS: Linux
* Docker installed
* [Dockerhub](https://hub.docker.com/) account
* Port 9999 open 

<a id='setup'></a>
## Setup 
Before you begin make sure you are logged into your dockerhub account by running on your machine:

```bash
docker login 
```
### Setup Batch AI Images
We need to create the images that will run our code on Batch AI. For chosen framework, you first navigate to its corresponding directory and then build the docker image. Taking TensorFlow model as an example, you navigate to [HorovodTF folder](./HorovodTF) and run following command to build the image (replace any instance of <dockerhub account> with your own dockerhub account name):

```bash
make build dockerhub=<dockerhub account>
```

Then push the image to your registry with:

```bash
make push dockerhub=<dockerhub account>
```

### Setup Execution Environment
Before being able to run anything you will need to configure your machine (local host) to set up the environment in which you will be executing the Batch AI commands etc. There are a number of dependencies therefore we offer a dockerfile that will take care of these dependencies for you. To build the image run (replace all instances of <dockerhub account> with your own dockerhub account name) following command in current directory:

```bash
make build dockerhub=<dockerhub account>
```
Then start the jupyter notebook on port 9999: 
```bash
make jupyter dockerhub=<dockerhub account>
```

By following the instructions shown in the output messages of above command, simply point your browser to the IP or DNS of your machine. From there you can navigate to the folders for tutorials on the frameworks covered such as HorovodTF etc.

Alternatively, if you don't want to use Docker, you can look inside the Docker directory at the dockerfile and environment.yml file for the dependencies.

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
