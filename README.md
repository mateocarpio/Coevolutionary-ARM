# The Coevolutionary Attraction-Repulsion Model 

The Coevolutionary Attraction-Repulsion Model with Mass Media is an agent-based model focused on opinion dynamics to study the polarization of a system. This model has been developed by Mateo Carpio and Mario Cosenza. 

This Python implementation simulates the ARM, allowing variations in the following parameters:

| Parameter     |  | Meaning    |
|----------|-----|-------------|
| Rewiring Probability | $P_r$  | Probability of having dynamics on the topology |
| Tolerance           | $T$ | Distance threshold within which two agents can get connected. Also distance within which interactions are attractive and beyond which interactions are repulsive|
| Exposure            | $E$  | Degree to which actors interact with differing points of view |
| Responsiveness      | $R$ | Fractional distance an actor's ideological position moves as a result of an interaction |
| Number of agents    | $N$ | Number of nodes that make up the network |

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

## 2) Getting_Results_Coevolutive_ARM.ipynb
In this notebook, we plot the results of the experiments. We can find the results for the asymptotic statistical quantities as a function of $P_r$ and $T$.

## 3) videos

This folder contains simulations of some remarkable examples in gif format. 
