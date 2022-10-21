# AIMI_PatchCam_Challenge

### Introduction
This repository holds the codebase for our contribution to the PatchCam Grand challenge. Our participation in this 
challenge was for the practical part of a course, 'AI for Medical Imaging', as part of the MAster program Artificial 
Intelligence at the University of Amsterdam. 

### Files
We experimented with three different models, GDenseNet, DenseNet, and ViT. 

- `keras_baseline_scripts` holds the code for reproducing our results for the GDenseNet model
- `models` holds the trained model files which can be used to load the models without training
- `predictions` contains the csv files with results
- `VIT old` contains old code for your first tryout for implementing the ViT

### Installing the environment
Use the YML file `environment.yml` to install the environment using Conda.

### Run the code
Use `main.py` to run the main training and testing scripts. Use the `settings.py` file to set hyperparameters and to 
specify the type of model and augmentation. 

For running the ViT also the seperate `ViT_pretrained.py` file can be used to run this model. 

