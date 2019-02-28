#################################################################
# Config file for a specific dataset
#
# Project name:
# ExplainableMS
# Fine-tuning
#
# add dataset specific parameters at the bottom
# 
#################################################################
import os

##### general parameters #####
# main project folder
root_dir = "/analysis/share/Ritter/Dataset/MS"
# data folder
data_dir = os.path.join(root_dir, "CIS")
doc_dir = os.path.join(root_dir, "Test", "file_list_HC_MS_BET_FLAIR.csv")

# uncomment if data contains NANs and you wish to remove them
# remove_nan = True
# set GPU ID to use (use 0 if only one available)
gpu = 0
# define shape of datapoints
shape = (96, 114, 96)

##### training parameters #####
# batch size
# i.e. number of samples per training step
# validation always uses b = 1
b = 4
# number of epochs
# i.e. times to loop over the entire set
num_epochs = 200 

##### dataset specifc parameters #####
indices_holdout = [ 99,  93,  54, 103,  98,  75,  89,  53,  44,  59,
        5,  16,  91, 14,  58,  33,  73,  29,  66,  35, 117,  84,  31]
