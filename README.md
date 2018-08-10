# Training Distributed Training on Batch AI

This repo is a tutorial on how to train a CNN model in a distributed fashion using Batch AI. 

## Prerequisites
* Linux
* Docker installed
* Dockerhub account
* Port 9999 open 

## Setup 
Before you begin make sure you are logged into your dockerhub account by running on your machine:

```bash
docker login 
```

### Setup Batch AI containers
Before we do anything we need to create the conatiners that will run our code on Batch AI. You can do this by navigating to one of the framwork folders such as HorovodTF and running(replace any instace of <dockerhub account> with your own dockerhub account name):

```bash
make build dockerhub=<dockerhub account>
```

Then push the container to your registry with:

```bash
make push dockerhub=<dockerhub account>
```

### Setup Execution Environment
Before being able to run anything you will need to set up the environment in which you will be executing the Batch AI commands etc. There are a number of dependencies therefore we offer a dockerfile that will take care of these dependecies for you. If you don't want to use Docker simply look inside the Docker directory at the dockerfile and environment.yml file for the dependencies. To build the container run(replace all instances of <dockerhub account> with your own dockerhub account name):

```bash
make build dockerhub=<dockerhub account>
```

```run
make jupyter dockerhub=<dockerhub account>
```

This will start the jupyter notebook on port 9999. Simply point your browser to the IP or DNS of your machine. From there you can navigate to the folders for tutorials on the frameworks covered such a HorovodTF etc.



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
