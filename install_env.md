# Guidelines to install a Conda virtual environment  

This document describes how to create a virtual environment in Anaconda, necessary to run the python codes of D-CASCADE (Dynamic Catchment Sediment Connectivity And Delivery). 

Anaconda distributes the Python language (so you do not have to download it separately) and also automatically manage library upgrades and dependencies according to your operating system (Windows, Linux etc.). It also allows you to install all the libraries in a virtual environment, so whatever you do there you will not harm your computer. 

## Copy the repository in your local computer 
Create a clone of this repository on your computer. This can be done using Git on the command line, e.g. running the following command

```console
git clone https://github.com/CHASM-UoB/stochastic-docs.git
```

> **NOTE:** If `git clone` fails with an error indicating that the `git` command cannot be found, then you may need to install Git on your computer.
> For instructions for installing Git on Linux, macOS, or Windows, see the section ["Installing Git"][pro_git_install] in the free online [Pro Git book][pro_git].



## Create a virtual environment with Anaconda

For D-cascade we created already a virtual envirnment that contains all the libraries you will need. 
This can be 

### 1) Install the Conda package manager

Creating a virtual environment requires the [Conda](https://conda.io/projects/conda/en/latest/index.html) package manager, which comes installed within Anaconda. If the conda command is already available on your computer (e.g. from an installation of the Anaconda distribution), you can skip to the next step. Otherwise, you can install conda by downloading and installing [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) for your operating system. Follow the instruction for installation. If prompted, check the box “Add Anaconda to my PATH environment”.

### 3) Create a conda environment 

After the installation is completed, you can look for the Anaconda prompt on the start menu.
Right click on it and open it as an administrator. 
