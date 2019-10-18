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
|   |   ├──MME
|   |   |   └── ...
|   |   ├── curves.png
|   |   └── curves.svg
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

This repository contains the code implementing the method described in the paper
"Unsupervised Emergence of Egocentric Spatial Structure from Sensorimotor
Prediction" (submitted to NeurIPS, 2019).


## Usage

All scripts should be run using Python 3.5. You will also need TensorFlow to be properly
installed on your computer to train the network (tested on tf versions 1.4 and 1.10).


To generate 50 sensorimotor datasets of 150000 transitions and save them in dataset/explo0, use:
```
generate_sensorimotor_data.py -n 150000 -t <type> -r 50 -d dataset/explo0
```
\<type\> can be one of the three strings:

* gridexplorer3d - for an agent with 3 motor DoF in a discrete gridworld (fast)
* gridexplorer6d - for an agent with 6 motor DoF in a discrete gridworld (fast)
* arm3droom - for a three-segment arm moving a RGB camera in a 3D environment (slow)


To train a network on these datasets with a motor encoding space of dimension 3 and save the models in model/experiment0, use:
```
train_network.py -dd dataset/explo0 -dm model/experiment0 -dh 3 -v -mem -mm -mme
```

To track the representation quality during training, use:
```
tensorboard --logdir=model/experiment0
```
and connect to the TensorBoard server.

To analyze the results of the training and save the corresponding curves in model/experiment0, use:
```
analyze_network.py -d model/experiment0
```


## Advanced control

For a finer control of the simulation, use:
```
generate_sensorimotor_data.py -n <number_transitions> -t <type> -r <number_datasets> -d <dataset_destination> -v <visualization_flag> -s <scaleup_MM_explo_flag>

train_network.py -dd <dataset_directory> -dm <model_destination> -dh <encoding_dimension> -e <number_epochs> -n <number_of_runs> -v <visualization_flag> -gpu <gpu_usage_flag> -mem <train_on_MEM_flag> -mm <train_on_MM_flag> -mme <train_on_MME_flag>

analyze_network.py -d <model_directory>
```
or check the scripts and provided help:
```
generate_sensorimotor_data.py -h

train_network.py -h

analyze_network.py -h
```