# dcascade-py

D-CASCADE (Dynamic CAtchment Sediment Connectivity And Delivery) is a modelling framework for sediment transport and connectivity analysis in large river networks (Tangi et al. (2022), Doolaeghe et al. (in prep)).
It is the dynamic version of the CASCADE framework (Schmitt et al. (2016)).
A new release (v2.0.0) was recently created (Doolaeghe et al. (in prep)).

This repository contains:
- all the core function necessary to run the model (src folder).
- one example of user script (user_scripts folder), that can be runned to test the model installation.
- examples of inputs to the model (inputs folder)
- examples of python script to post-process and analyse the results

The model is written in python.

Developers: Diane Doolaeghe, Anne-Laure Argentin, Elisa Bozzolan, Felix Pitscheider


# Installation

## Downloading the code

On repository page, find the Release section and click on v2.0.0. Then click on "Source code (zip)". This normally starts downloading your project, that should be placed in your download folder.
Copy-paste the project somewhere convenient on your computer.

## Instructions for installing the conda environment

To respect the requirements of D-CASCADE (i.e. correct python and packages versions), we propose you to install a conda environment to run D-CASCADE on your computer.
This environment is stored in the file "environment.yml", which is inside the D-CASCADE repository.

To install this environment, you will need to have Anaconda or Miniconda installed on your computer. If not, please install [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html), for your operating system.
Note: Anaconda distributes the Python language (so you do not have to download it separately) and automatically manages the python libraries upgrades and dependencies according to your operating system (Windows, Linux etc.). It also allows you to install all the libraries in a virtual environment to avoid any potential damage to your computer.
Creating a virtual environment requires the [Conda](https://conda.io/projects/conda/en/latest/index.html) package manager, which normally comes installed within Anaconda.

Once Anaconda is installed on your computer, you can look for "Anaconda Prompt" on the start menu.
Right click on it and open it as an administrator.
Then, navigate to the path where your environment.yml file is stored, which is where you have installed your D-CASCADE project on your computer:

```console
cd name_of_the_path
```

To check whether the "environment.yml" file is in there you can type: `dir` and it should appear.

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

To check if the installation went well, you can use the example, that is available on the repository (using one small river network of the Vjosa river).
Open the user script example, available at "user_scripts\00-DCASCADE_user_script_example_Vjosa.py", in Spyder (you can drag it into Spyder). And run it.
You should see a time bar progressing quickly in the Spyder console.

Once done, the simulation should create a folder "cascade_results" in your project, and produce two outputs files (Vjosa_test.p and Vjosa_test_exp.p).
These are pickle files containing all outputs of the model.

## First checking of outputs

In the folder "post_process_examples", there is a list of python script, that are examples on how to post-process the results, and make graphs:

- scripts that show how to analyse each reach sediment transport along time (total, per grain size, and by initial provenance, (01, 02, 03))

- scripts that show how to analyse yearly sediment transport along the river network (total yearly sum, per grain size, and by initial provenance, (04, 05, 06))

- a script to create a dynamic map of the outputs (07)

- a script to analyse connectivity and sediment path-length per time step (08)

You can run them and check the type of plot they generate for the example case. 


## Making your own project

You can modify the user_script example (00-DCASCADE_user_script_example_Vjosa.py) to correspond to your own project.

Inputs to the model (reachdata and discharge) are generated externally. Examples are provided in the folder "inputs", and instruction for generating them are available at: TODO.
In your user script, modify the path to point at your input location.

Modify also the output name to store your own outputs in the "cascade_results" folder.

You can modify the simulation parameters to correspond to your river network, and then run your case study. 

Modify also the paths in the post-process example scripts, so that they coincide with your output file name in the "dcascade_result" folder. 


# Documentation of D-CASCADE code functions
https://dcascade-py.github.io/dcascade-py-doc/index.html
