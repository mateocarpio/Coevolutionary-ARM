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
                for step in trange(S, desc='Simulating interactions', disable=True):
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

class process_data():

    def __init__(self, Pr, T, iters, hist_saved, max_steps):

        self.Pr = Pr

        self.T = T

        self.iters = iters

        self.directory_name = "Pr_{:.5f}-T_{:.2f}".format(round(Pr, 5), round(T, 2))

        self.hist_saved = hist_saved

        self.max_steps = max_steps

        delta_bins = 1/50
        
        self.bins = np.arange(0,1+delta_bins,delta_bins)

    def create_subplots(self, G, pos, step):

        config=[]

        for k in G.nodes:

            config.append(G.nodes[k]["opinion"][0])

        # Create subplots
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        delta_bins = 1/50
        bins = np.arange(-delta_bins/2, 1+delta_bins, delta_bins)

        # Define the number of bins and the color map
        num_bins = len(bins)
        color_map = plt.cm.get_cmap('coolwarm')
        norm = mcolors.Normalize(vmin=0, vmax=num_bins-1)

        # Create the histogram
        n, bins, patches = ax[0].hist(config, bins, edgecolor='black')

        # Set the color of each patch based on its position in the histogram
        for i in range(len(patches)):
            color = color_map(norm(i))
            patches[i].set_facecolor(color)

        # Set the axis labels and title
        ax[0].set_xlabel('Opinion Position', size=12)
        ax[0].set_ylabel('Total Agents', size=12)
        ax[0].set_xlim([-0.02, 1.02])
        ax[0].set_ylim([0, 101])
        ax[0].set_title('Step %d' % step, size=12)

        ax[0].tick_params(axis='both', which='major', labelsize=12)
        ax[0].tick_params(axis='both', which='minor', labelsize=12)

        new_pos = nx.spring_layout(G, seed=38)

        for node in pos:

            pos[node] += (new_pos[node] - pos[node]) * 0.1

        # Get the node colors from their 'opinion' attributes
        node_colors = [node[1]['opinion'] for node in G.nodes(data=True)]

        nx.draw(G, pos=pos, node_color=node_colors, cmap='coolwarm',
                with_labels=False, node_size=100, ax=ax[1], width=0.5, vmin=0, vmax=1)

        ax[0].set_title('Step %d' %step, size = 12)

        if not os.path.exists("./outputfolder/"+str(self.directory_name)+"/figures"):
            os.makedirs("./outputfolder/"+str(self.directory_name)+"/figures")

        plt.savefig("./outputfolder/"+ str(self.directory_name) +"/figures/{:03d}.png".format(step), bbox_inches='tight', pad_inches=0.1)

        plt.show()

        return pos

    def create_images(self):

        with open(f"./outputfolder/{self.directory_name}/networks/network_iter-{self.iters}_step-{0}", "rb") as file:

              G = pickle.load(file)

        pos = nx.spring_layout(G, scale=2, seed=213123)

        for step in range(0,self.last_step+1,500):

            try:
              with open(f"./outputfolder/{self.directory_name}/networks/net_iter-{self.iters}_step-{step}", "rb") as file:

                G = pickle.load(file)

                pos = self.create_subplots(G, pos, step)

            except:

              break

   #Identify the size of the maximum group and where it is located
    def maximum_group(self, frequency):

        m = max(frequency)

        indexes=[i for i, j in enumerate(frequency) if j == m]

        return m,indexes


   #Identify the size of the maximum group and where it is located
    def central_maximum_group(self, frequency):

        central_frequency = frequency[12:-12]

        m = max(central_frequency)

        indexes=[i for i, j in enumerate(frequency) if j == m]

        return m, indexes



    #Get the variance, the maximum group and its difference with the MM
    def get_statistical_parameters(self, config):

        frequency, bins = np.histogram(config,bins = self.bins)

        SM, indexes = self.maximum_group(frequency)

        cental_SM, indexes = self.central_maximum_group(frequency)

        variance = np.var(config)

        return cental_SM, SM, variance

    # Plot the statistical parameters
    def save_stat(self):

        file_path = "./history_statistics/statistical_parameters_Pr_{:.5f}-T_{:.2f}.txt".format(round(self.Pr, 5), round(self.T, 2))
        if not os.path.exists("./history_statistics"):

            os.makedirs("./history_statistics")

        with open(file_path, "w") as f:

            f.write("Step \t Central-S_max \t S_max  \t variance \n")

        for step in range(0,self.max_steps+1,100):
          
          try:
            file_path_network = "./outputfolder/"+ str(self.directory_name) + "/networks" + f"/net_iter-{self.iters-1}_step-{step}"

    
            with open(file_path_network, "rb") as file:

                  G = pickle.load(file)

                  config=[]

                  for k in G.nodes:

                      config.append(G.nodes[k]["opinion"][0])

                  c_SM, SM, variance = self.get_statistical_parameters(config)

                  with open(file_path, "a") as f:

                      f.write("{:.1f}\t{:.6f}\t{:.6f}\t{:.6f}\n".format(step, c_SM, SM, variance))
          except:
            
            pass

    #Get the variance, the maximum group and its difference with the MM
    def get_asymptotical_parameters(self, it):
        
        C_SM, SM, variance = 0, 0, 0
        counter=0
        for step in range(0,self.max_steps,100):
            try:
                file_path = "./outputfolder/"+ str(self.directory_name) + "/networks" + f"/asympNet_iter-{it}_step-{step}"
                print(file_path)
                with open(file_path, "rb") as file:
                    G=pickle.load(file)
                    config=[]
                    for k in G.nodes:
                        config.append(G.nodes[k]["opinion"][0])
                    counter = counter + 1    
                    new_c_SM, new_SM, new_variance = self.get_statistical_parameters(config)
                    C_SM, SM, variance = C_SM + new_c_SM, SM + new_SM, variance + new_variance
            except:

                pass
        C_SM, SM, variance = C_SM/counter, SM/counter, variance/counter 
        return  C_SM, SM, variance 

    def prom_iters(self):

        prom_C_SM, prom_SM, prom_variance= 0, 0, 0

        for it in range(self.iters):

            iter_C_SM, iter_SM, iter_variance = self.get_asymptotical_parameters(it)

            prom_C_SM, prom_SM, prom_variance = prom_C_SM + iter_C_SM, prom_SM + iter_SM, prom_variance + iter_variance

        return prom_C_SM/self.iters, prom_SM/self.iters, prom_variance/self.iters

    def save_prom_parameters(self):

        file_path = "./Asymptotic_statistics/Asymp_statistical_parameters_Pr_{:.5f}-T_{:.2f}.txt".format(round(self.Pr, 5), round(self.T, 2))

        if not os.path.exists("./Asymptotic_statistics"):

            os.makedirs("./Asymptotic_statistics")

        with open(file_path, "w") as f:

            f.write("Iteration \t C_S_max  \t S_max  \t sigma \n")

        prom_C_SM, prom_SM, prom_variance= 0, 0, 0

        for it in range(self.iters):

            iter_C_SM, iter_SM, iter_variance = self.get_asymptotical_parameters(it)

            with open(file_path, "a") as f:

                f.write("{:.1f}\t{:.6f}\t{:.6f}\t{:.6f}\n".format(it, iter_C_SM, iter_SM, iter_variance))

            prom_C_SM, prom_SM, prom_variance = prom_C_SM + iter_C_SM, prom_SM + iter_SM, prom_variance + iter_variance

        with open(file_path, "a") as f:

            f.write("average\t{:.6f}\t{:.6f}\t{:.6f}\n".format(prom_C_SM/self.iters, prom_SM/self.iters, prom_variance/self.iters))


#Reproduces Figures 4.4 and 4.5
def run_history(Pr, T):
    params = {'N' : [100], 'E' : [0.1], 'T' : [T], 'R' : [0.25], 'S' : [2000001], 'Pr' : [Pr]}
    exp = ARM_Coevolution(params, iters=1, seed=None, savehist=True)
    exp.arm_coe()
    results = process_data(Pr, T, iters=1, hist_saved=True, max_steps=2000001)
    results.save_stat()
def exp_history_Pr_T(seed, n_cpu): #for parallelize
    pool = mp.Pool(processes=n_cpu)
    # Define ranges of B and X_M values
    Pr_range = [0.00, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    T_range = [0.20, 0.25]
    # Create a list of tuples containing all combinations of XM and B values
    param_tuples = [(Pr, T) for Pr in Pr_range for T in T_range]
    # Call run_asymptotic_SW for all parameter tuples using pool.map
    results = pool.starmap(run_history, param_tuples)
    # Close the pool

#Reproduces Figures 4.6
def run_asymptotic(Pr, T):
    params = {'N' : [100], 'E' : [0.1], 'T' : [T], 'R' : [0.25], 'S' : [2000001], 'Pr' : [Pr]}
    exp = ARM_Coevolution(params, iters=2, seed=None, savehist=False)
    exp.arm_coe()
    results = process_data(Pr, T, iters=2, hist_saved=False, max_steps=2000001)
    results.save_prom_parameters()
def exp_Pr_T(seed, n_cpu): #for parallelize
    # Call Pool
    pool = mp.Pool(processes=n_cpu)
    # Define ranges of B and X_M values
    Pr_range = np.logspace(np.log10(0.0001), np.log10(0.9), num=100)
    T_range = [0.10, 0.15, 0.20, 0.25]
    # Create a list of tuples containing all combinations of XM and B values
    param_tuples = [(Pr, T) for Pr in Pr_range for T in T_range]
    # Call run_asymptotic_SW for all parameter tuples using pool.map
    results = pool.starmap(run_asymptotic, param_tuples)
    # Close the pool
    pool.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('-n', '--ns', type=int, default=2, \
        help="-takes the number of processes to be used in parallelization",)
    parser.add_argument('-E', '--exps', type=str, nargs='+', required=True, \
                        help='IDs of experiments to run')
    parser.add_argument('-R', '--rand_seed', type=int, default=None, \
                        help='Seed for random number generation')
    args = parser.parse_args()
    # Run selected experiments.
    exps = {'exp_Pr_T' : exp_Pr_T, 'exp_history_Pr_T' : exp_history_Pr_T}

    for id in args.exps:
        exps[id](args.rand_seed,args.ns)
