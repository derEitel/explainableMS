#################################################################
# Config file for a specific dataset
#
# Project name:
# ExplainableMS
# Pretraining on Alzheimer's disease dataset (ADNI)
#
# add dataset specific parameters at the bottom
# 
#################################################################
import os

##### general parameters #####
# main project folder
root_dir = "/analysis/share/ADNI/"
# data folder
data_dir = os.path.join(root_dir, "HDF5_files")

# uncomment if data contains NANs and you wish to remove them
# remove_nan = True
# set GPU ID to use (use 0 if only one available)
gpu = 0
# define shape of datapoints
shape = (96, 114, 96)

##### training parameters #####
# number of folds
k_folds = 7
# batch size
# i.e. number of samples per training step
# validation always uses b = 1
b = 8 
# number of epochs
# i.e. times to loop over the entire set
num_epochs = 45

##### dataset specifc parameters #####
