# dcascade-py

D-CASCADE (Dynamic CAtchment Sediment Connectivity And Delivery) is a modelling framework for sediment transport and connectivity analysis in large river networks (Tangi et al. (2022), Doolaeghe et al. (in prep)).
It is the dynamic version of the CASCADE framework (Schmitt et al. (2016)).
A new release (v2.0.0) was recently created (Doolaeghe et al. (in prep)).

This repository contains:
- all the core function necessary to run the model (src folder).
- one example of user script (user_scripts folder), that can be runned to test the model installation.
- examples of inputs to the model (inputs folder)

The model is written in python.

Developers: Diane Doolaeghe, Anne-Laure Argentin, Elisa Bozzolan, Felix Pitscheider


# Installation

You have two option, depending if you which to version your project under Github or not.
Github allows you to track you developments and eventually, later on, add them to the main version (after agreement with the Padova team).

## Installation WITHOUT Github versionning

On repository page, find the Release section and click on v2.0.0. Then click on "Source code (zip)". This normally starts downloading your project, that should be placed in your download folder.
Copy-paste the project somewhere convenient on your computer.

## Installation WITH Github versionning

--> TODO



## Instructions for installing the conda environment

We recommand to use python 3.12. Additionnaly, some python packages must be installed before a given version (e.g. pandas 2.2).
To respect these requirements we propose you to install a conda environment to run D-CASCADE on your computer.

You will need to have Anaconda or Miniconda installed on your computer. If not, please install [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html), for your operating system.
Note: Anaconda distributes the Python language (so you do not have to download it separately) and automatically manages the python libraries upgrades and dependencies according to your operating system (Windows, Linux etc.). It also allows you to install all the libraries in a virtual environment to avoid any potential damage to your computer.

We have created a virtual environment that contains all the libraries you will need. It is stored in the file environment.yml, which is inside the repository.

Creating a virtual environment requires the [Conda](https://conda.io/projects/conda/en/latest/index.html) package manager, which normally comes installed within Anaconda.

Once Anaconda is installed on your computer, you can look for "Anaconda Prompt" on the start menu.
Right click on it and open it as an administrator.
Then, navigate to the path where your environment.yml file is stored, which is where you have installed your D-CASCADE project :

```console
cd name_of_the_path
```

The change should be visualised in brackets in your conda shell. To check whether the cascade.yml file is in there you can type: `dir` and it should appear.

Then, create the environment (called here "dcascade") with the required python version and packages. This may take a few minuts. 

```console
conda env create -f environment.yml -n dcascade
```

And activate it:

```console
conda activate dcascade
```

If the environment installation from the "environment.yml" file does not work, or is anormally to long, try these lines one after the other: 

```console
conda env create -n dcascade python=3.12.3
conda activate dcascade
```

```console
conda install spyder
conda install numpy
conda install tqdm
conda install matplotlib
conda install pandas=2.3.1
conda install networkx=3.5
conda install -c conda-forge geopandas=1.1.1
conda install -c conda-forge shapely=2.0.5
```


Now you can call spyder, or open it directly from your start menu (spyder(dcascade)).
```console
spyder
```
Spyder is an interpreter where you can visualise and run the python scripts of D-CASCADE


## First D-CASCADE run

To check if the installation went well, you can use the example, that is ready on the repository (using one small river network of the Vjosa river).
Open the user script example, available at "user_scripts\00-DCASCADE_user_script_example_Vjosa.py", in Spyder (you can drag it into Spyder). And run it.
You should see a time bar progressing quickly in the Spyder console.

Once done, the simulation should create a folder "cascade_results" in your project, and produce two outputs files (save_all.p and save_all_exp.p).
These are pickle files containing all outputs of the model.

## First checking of outputs

--> TODO
--> propose output processing examples


## Making your own project

You can modify the user_script example to correspond to your own project.
If you are under Github versionning, please duplicate in the same folder the example user script (00-DCASCADE_user_script_example_Vjosa.py), so that it remains untouched.

Inputs to the model (reachdata and discharge) are generated externally. Examples are provided in the folder "inputs", and instruction for generating them are available at: TODO.
In your user script, modify the path to point at your input location.
Modify also the output path to store your own outputs in the dcascade_results folder (TODO).

You can modify the simulation parameters to correspond to your river network.


# Documentation of D-CASCADE code functions
https://dcascade-py.github.io/dcascade-py-doc/index.html
