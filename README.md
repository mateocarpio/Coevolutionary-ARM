# The Coevolutionary Attraction-Repulsion Model 

The Coevolutionary Attraction-Repulsion Model with Mass Media is an agent-based model focused on opinion dynamics to study the polarization of a system. This model has been developed by Mateo Carpio and Mario Cosenza. 

## 1) coevolution_arm.py

This file contains the main code for running the Coevolutionary ARM. The code uses the multiprocessing library to parallelize the code across multiple processors.
The code has been written in Python and requires the following packages to be installed:
   
    numpy
    networkx
    multiprocessing
    math
    os
    itertools
    pickle
    product
    argparse
    

## Usage

The script can be run on a cluster or on a multi-core machine using the following command:

    python coevolution_arm.py -n <number_of_processes> -E <id_of_experiment> -R <random_seed>

where <number_of_processes> is the number of processes to be used in parallelization, <id_of_experiment> is the experiment wanted to run, and <random_seed> is the seed.

The output of the code is saved in the current and output folder directory, which is created if it does not exist.

## 3) videos

This folder contains simulations of some remarkable examples in gif format. 
