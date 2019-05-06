# Unsupervised Emergence of Egocentric Spatial Structure from Sensorimotor Prediction

## Author and Contact
Anonymous (anonymous@anonymous.com)

## Structure
```
.
├── dataset
|   ├── explo0
|   |   ├── dataset0
|   |   |   ├── agent.pkl
|   |   |   ├── agent_params.txt
|   |   |   ├── dataset_MEM.pkl
|   |   |   ├── dataset_MM.pkl
|   |   |   ├── dataset_MME.pkl
|   |   |   ├── environment.pkl
|   |   |   ├── environment_params.txt
|   |   |   └── uuid.txt
|   |   ├── dataset1
|   |   |   └── ...
|   |   └── ...
|   ├── explo1
|   └── ...
├── flatland
|   └── ...
├── gqn_renderer
|   └── ...
├── model
|   ├── experiment0
|   |   ├── MEM
|   |   |   ├── run0
|   |   |   |   ├── display_progress
|   |   |   |   |   └── ...
|   |   |   |   ├── model
|   |   |   |   |   ├── checkpoint
|   |   |   |   |   ├── model.ckpt.data-00000-of-00001
|   |   |   |   |   ├── model.ckpt.index
|   |   |   |   |   └── model.ckpt.meta
|   |   |   |   ├── tb_logs
|   |   |   |   |   └── ...
|   |   |   |   ├── network_params.txt
|   |   |   |   └── training_dataset_uuid.txt
|   |   |   ├── run1
|   |   |   └── ...
|   |   ├── MM
|   |   |   └── ...
|   |   └── MME
|   |   |   └── ...
|   ├── experiment1
|   |   └── ...
|   └── ...
├── .gitignore
├── Agents.py
├── analyze_network.py
├── display_progress.py
├── Environments.py
├── generate_sensorimotor_data.py
├── LICENSE
├── Network.py
├── README.md
├── requirements.txt
└── train_network.py
```


## Introduction

This repository contains a code implementing the method described in the paper
"Unsupervised Emergence of Egocentric Spatial Structure from Sensorimotor
Prediction" (submitted to NeurIPS, 2019).


## Usage

All scripts should be run using Python 3.5.


To generate a sensorimotor dataset and save it in ./dataset/explo0, use:
```
python3 generate_sensorimotor_data.py -t <type> -d dataset/explo0
```
\<type\> can be one of the three values:

* gridexplorer - for an agent in a discrete gridworld
* armflatroom - for a three-segment arm moving a distance sensor array in a flat environment 
* arm3droom - for a three-segment arm moving a RGB camera in a 3D environment


To train a network on this dataset and save the model in ./model/experiment0, use:
```
python3 train_network.py -dd dataset/explo0 -dm model/experiment0 -v -mem -mm -mme
```

To analyze the results of the training, use:
```
python3 analyze_network.py -d model/experiment0
```


## Advanced control

For a finer control of the simulation, use:
```
python3 generate_sensorimotor_data.py -n <number_transitions> -t <type> -r <number_datasets> -d <dataset_destination> -v <visualization_flag>

python3 train_network.py -dd <dataset_directory> -dm <model_destination> -dh <encoding_dimension> -e <number_epochs> -n <number_of_runs> -v <visualization_flag> -gpu <gpu_usage_flag> -mem <train_on_MEM_flag> -mm <train_on_MM_flag> -mme <train_on_MME_flag>

python3 analyze_network.py -d <model_directory>
```
or check the scripts and provided help:
```
python3 generate_sensorimotor_data.py -h

python3 train_network.py -h

python3 analyze_network.py -h
```