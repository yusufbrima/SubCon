#!/bin/bash

# Define SSH connection details
remote_host="voxel"

# Define the path and Conda environment name
remote_path="/net/store/cv/users/ybrima/RTGCompCog/SubCon"
conda_env_name="subcon"

# ssh into the remote machine and execute the commands
# ssh -y "$remote_host"

# Activate the Conda environment
conda activate "$conda_env_name"

# cd into the desired directory
cd "$remote_path"