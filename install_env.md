# Guidelines to install a Conda virtual environment  

This document describes how to create a virtual environment in Anaconda, necessary to run the python codes of D-CASCADE (Dynamic Catchment Sediment Connectivity And Delivery). 

Anaconda distributes the Python language (so you do not have to download it separately) and automatically manages the python libraries upgrades and dependencies according to your operating system (Windows, Linux etc.). It also allows you to install all the libraries in a virtual environment to avoid any potential damage to your computer. 

For D-cascade we have already created a virtual environment that contains all the libraries you will need. 
This environment is stored in the file cascade.yml which is inside the repository. 
This tutorial will go through the steps necessary to install the environment cascade.yml.

## Copy the repository in your local computer 
Create a clone of this repository on your computer. This can be done using Git on the command line, e.g. running the following command

```console
git clone https://github.com/elisabozzolan/dcascade_py.git
```

> **NOTE:** If `git clone` fails with an error indicating that the `git` command cannot be found, then you may need to install Git on your computer.
> For instructions for installing Git on Linux, macOS, or Windows, see the section [Installing git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) in the free online [Pro Git book](https://git-scm.com/book/en/v2). Once install, you can open a git bash and run in there the `git` command.


## Create a virtual environment with Anaconda

### 1) Install the Conda package manager

Creating a virtual environment requires the [Conda](https://conda.io/projects/conda/en/latest/index.html) package manager, which comes installed within Anaconda. If the conda command is already available on your computer (e.g. from an installation of the Anaconda distribution), you can skip to the next step. Otherwise, you can install conda by downloading and installing [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) for your operating system. Follow the instruction for installation. If prompted, check the box “Add Anaconda to my PATH environment”.

### 3) Create a conda environment 

After the installation is completed, you can look for the Anaconda prompt on the start menu.
Right click on it and open it as an administrator. 
Navigate in the path where your cascade.yml file is stored (which will be where you cloned the repository) 

```console
cd name_of_the_path
```
The change should be visualised in brackets in your conda shell. To check whether the cascade.yml file is in there you can type: `dir` and it should appear. 

To install the `cascade` environment type:  
```console
conda env create -f cascade.yml
```
It might take several minutes...

To check that the environment has been successfully created, activate it:
```console
conda activate cascade
```

Now you can call 
```console
spyder
```
where you can visualise and run the python scripts of D-CASCADE 


