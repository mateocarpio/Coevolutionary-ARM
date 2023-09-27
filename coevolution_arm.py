#!/usr/bin/env python

#import libraries
import networkx as nx
import numpy as np
from tqdm import trange
import math
from itertools import product
import os
import pickle
import multiprocessing as mp

class ARM_Coevolution():
    def __init__(self, params, iters, seed, savehist=True):
        defaults = {'N' : [100], 'E' : [0.1], 'T' : [0.25], 'R' : [0.25], 'S' : [500000], 'Pr' : [0.5]}
        plist = [params[p] if p in params else defaults[p] for p in defaults]
        self.params = list(product(*plist))
        self.iters = iters
        self.rng = np.random.default_rng(seed)
        self.savehist = savehist
        self.directory_name = "Pr_{:.5f}-T_{:.2f}".format(round(plist[5][0], 5), round(plist[2][0], 2))

    #Create intial opinions
    def initializing(self, N):
        config = np.zeros(N)
        for i in np.arange(N):
        #initial opinions follows a Gaussian distribution
            while True:
                config[i] = self.rng.normal(0.5, 0.2)
                if 0 <= config[i] and config[i] <= 1:
                    break
        config = config.reshape(-1, 1)
        init_config = config
        return config

    #Create the random network    
    def random_network(self, N, config):
        G = nx.random_regular_graph(4, N)
        for i in G.nodes:
                G.add_nodes_from([i], opinion=config[i])
        return G

    #Save the network
    def save_network(self, G, step, directory, it):
        if not os.path.exists("./outputfolder/"+str(directory)+"/networks"):
          os.makedirs("./outputfolder/" + str(directory)+"/networks")
        with open(f"./outputfolder/{directory}/networks/net_iter-{it}_step-{step}", 'wb') as f:
            pickle.dump(G, f)

    #Save the asymptotics networks
    def save_asymp_network(self, G, step, directory, it):
        if not os.path.exists("./outputfolder/"+str(directory)+"/networks"):
          os.makedirs("./outputfolder/" + str(directory)+"/networks")
        with open(f"./outputfolder/{directory}/networks/asympNet_iter-{it}_step-{step}", 'wb') as f:
            pickle.dump(G, f)

    #stop conditon        
    def test_if_can_stop(self, G, vars):
        opinions = nx.get_node_attributes(G, "opinion").values()
        variance = np.var(list(opinions))
        vars.append(variance)
        _res = False
        _reason = None
        counter = 0
        if len(vars) > 100:
            for i in range(100):
                if abs(vars[-1] - vars[-(i+1)]) < 10**(-6):
                    counter +=1
            if counter == 100:
                _res = True
                _reason = 'No variance change'
        return _res, _reason, vars

    #Dynamics based on the attraction-repulsion rule
    def node_dynamics(self, G, i, T, R, E):
        if len(G[i])>0:
            j = self.rng.choice(G[i]) #choose a random neighbor
            dist = (abs(G.nodes[i]["opinion"] - G.nodes[j]["opinion"])) #calcualte distance between opinions
            prob = math.pow(0.5, dist/E)
            if self.rng.random() <= prob:
                if dist <= T: #condition for atrarction d < T
                  #i get closer to j R times their distance
                    G.nodes[i]["opinion"] = G.nodes[i]["opinion"] + R * (G.nodes[j]["opinion"] - G.nodes[i]["opinion"])
                else: #condition for repulsion d > T
                    G.nodes[i]["opinion"] = G.nodes[i]["opinion"] - R * (G.nodes[j]["opinion"] - G.nodes[i]["opinion"])
                G.nodes[i]["opinion"] = np.maximum(0, np.minimum(1,G.nodes[i]["opinion"])) #set limits [0-1]
        return G

    #Disconnect of any neighbor 
    def disconect(self, G, i, T):
        condition = False
        choices = []
        #To prevent isolated nodes
        for k in G[i]:
          if len(G[k])>1:
            choices.append(k)
        if len(choices) > 0:
            j = self.rng.choice(choices)
            G.remove_edge(i,j)
            condition = True
        return G, condition

    #Connect to a neighbor that is incide the confidence bound
    def rewiring(self, G, i, T):
      # To control if there is possible the rewiring
        condition = False
        possible_choices = list(set(G.nodes) - set(G.neighbors(i)) - set({i})) #No neighbors
        set_choice = []
        #Select neighbors between the tolerance T
        for k in possible_choices:
            if abs(G.nodes[i]["opinion"] - G.nodes[k]["opinion"]) < T:
                set_choice.append(k)
        if len(set_choice) > 0:
            condition = True
            l = self.rng.choice(set_choice)
            G.add_edge(i, l)
        return condition, G

    def perform_time_step(self, G, T, R, E, pr):
        i = self.rng.choice(G.nodes) #choose a random node
        if self.rng.random() <= pr:
            condition_1, G = self.rewiring(G, i, T)
            if condition_1 == True:
              G, condition_2 = self.disconect(G, i, T)
              if condition_2 == False:
                G.remove_edge(i, l)
        else:
            G = self.node_dynamics(G, i, T, R, E)
        return G

    def arm_coe(self):
        for param in self.params:
            print(param)
            N, E, T, R, S, Pr = param
            directory_name = "Pr_{:.5f}-T_{:.2f}".format(round(Pr, 5), round(T, 2))
            if not os.path.exists("./outputfolder/"+str(directory_name)):
                os.makedirs("./outputfolder/" + str(directory_name))
            for it in range(self.iters):
                config = self.initializing(N)
                G = self.random_network(N, config)
                pos = nx.spring_layout(G, scale=2, seed=213123)
                vars = []
                _res = False
                inner_loop_terminated = False
                for step in trange(S, desc='Simulating interactions', disable=False):
                    G = self.perform_time_step(G, T, R, E, Pr)
                    if step%N==0:
                      if self.savehist == True:
                        self.save_network( G, step, directory_name, it)
                     #evaluete the stop condition 
                      _res, _reason, vars = self.test_if_can_stop(G, vars)
                      if _res == True:
                            self.save_asymp_network(G, step, directory_name, it)
                            inner_loop_terminated = True
                            self.save_asymp_network(G, step, directory_name, it)
                            break
                    #save the last 1000 steps
                    if step >= S-1000:
                        self.save_asymp_network(G, step, directory_name, it)
        return G, step


def expA_Pr_T(Pr=0.1, T=0.25):
    params = {'N' : [100], 'E' : [0.1], 'T' : [T], 'R' : [0.25], 'S' : [2000001], 'Pr' : [Pr]}
    exp = ARM_Coevolution(params, iters=50, seed=None, savehist=False)
    exp.arm_coe()

if __name__ == "__main__":
    #Number of laptop CPUs to be used
    n_cpu = 1
    # Call Pool
    pool = mp.Pool(processes=n_cpu)
    # Define ranges of XM and B values
    Pr_range = np.logspace(np.log10(0.0001), np.log10(0.9), num=100)
    T_range = [0.20,0.25]
    # Create a list of tuples containing all combinations of XM and B values
    param_tuples = [(Pr, T) for Pr in Pr_range for T in T_range]
    # Call expC_grid for all parameter tuples using pool.map
    results = pool.starmap(expA_Pr_T, param_tuples)
    # Close the pool
    pool.close()
