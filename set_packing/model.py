import logging
import numpy as np
from gym import spaces
import gym
from gym.utils import seeding
from csv import *
import random
import networkx as nx
import operator

logger = logging.getLogger(__name__)

class Set_packing(gym.Env):
    '''
    Dynamic set packing environment
    '''
    def __init__(self):
        self.N = 10 # universe elements
        self.k = 3 # number of elements in a subset
        self.l = 10 # universe subsets
        self.universe_graph = nx.Graph()
        self.current_elements = set() # record the current vertices in the pool
        self.current_subsets = []
        self.current_graph = nx.Graph()
        self.arrive = 3 # mean of elements arriving distribution
        self.depart = 1 # mean of elements depart distribution

        self.record = False
        self.load = False

        self.episode_length = 10  # length of each episode
        self.time = 1  # time tic in one episode
        self.done = False  # indicate episode over
        self.episode = 0  # record the current episode number
        self.reward = 0.0  # current reward
        self.total_rewards = 0.0
        self.all_rewards = 0.0
        self.agent = 'policy'


    def setup(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = self.embedding.observation_space
        self._seed()
        self.univese_pool = np.zeros([self.l, self.k])  # universe pool
        self.universe_matrix = np.zeros([self.l, self.l])
        self.path = '../data/dataN%sk%sl%s' % (self.N, self.k, self.l)
        self.log_path = '../logfile/log%sN%sk%sl%s' % (self.agent, self.N, self.k, self.l)
        self.generate_data()



    def generate_data(self):
        '''
        Generate or load the universe pool and graph based N, k, l and save it into the file
        '''
        if self.load == False:
            for i in range(self.l):
                self.univese_pool[i,:] = random.sample(range(self.N), self.k)
            np.savetxt(self.path,self.univese_pool,delimiter=",")

        else:
            self.univese_pool = np.loadtxt(fname= self.path,delimiter=",")

        # construct the universe connection graph
        for i in range(self.l):
            subset1 = set(self.univese_pool[i,:])
            for j in range(self.l):
                if j == i:
                    continue
                subset2 = set(self.univese_pool[j,:])
                if subset2.isdisjoint(subset1) == False:
                    self.universe_matrix[i,j] = 1


        self.universe_graph = nx.Graph(self.universe_matrix)


    def elements_to_graph(self):
        '''
        Convert the current elements into a sub_graph based on the universe graph
        '''
        stats = np.zeros([self.l, self.k])

        for i in self.current_elements:
            stats[self.univese_pool==i] = 1

        self.current_subsets = np.where(np.sum(stats, axis=1) >= self.k)[0].tolist()
        self.current_graph = self.universe_graph.subgraph(self.current_subsets)

    def add_elements(self, arrive):
        '''
        Sample elements from global and add into the current elements
        '''
        self.add_number = abs(int(np.random.normal(arrive, 8)))
        # print('add number',self.add_number)
        if self.add_number >= self.N:
            add_elements = set(range(self.N))
        else:
            add_elements = set(random.sample(range(self.N), self.add_number))
        self.current_elements = self.current_elements.union(add_elements)

    def depart_elements(self):
        '''
        Sample elements from current pool and add delete them
        '''
        self.del_number = abs(int(np.random.normal(self.depart, 0.1)))
        # print ('del number',self.del_number)
        if self.del_number >= len(self.current_elements):
            self.current_elements.clear()
        else:
            self.del_elements = set(random.sample(self.current_elements,self.del_number))
            self.current_elements = self.current_elements - self.del_elements

    def _seed(self, seed=None):
        self.rng, seed = seeding.np_random(seed)
        self.rng_seed = seed
        return [seed]

    def MIS_solver(self):
        '''
        Given a graph, solve a Maximum Independent Set by greedy algorithm
        Return: new self.current_elements and new self.current_graph
        '''
        graph = self.current_graph
        match_nodes = []
        reward = 0
        # print(nx.to_numpy_matrix(graph))
        list_nodes = graph.degree()
        # print('nodes degree',list_nodes)

        while len(list_nodes) > 0:
            min_degree = min(list_nodes.items(), key=operator.itemgetter(1))
            match_nodes.append(min_degree[0])
            reward += self.k # add the reward for each matched node
            neighbors = nx.all_neighbors(graph, min_degree[0])
            graph.remove_node(min_degree[0])
            graph.remove_nodes_from(neighbors)
            list_nodes = graph.degree()

        # Based on the match_nodes, delete the corresponding elements in self.current_elements
        match_elements = self.univese_pool[match_nodes,:].reshape((-1,)).tolist()
        # print('match elements',sorted(match_elements))
        # print('pre',self.current_elements)
        self.current_elements = self.current_elements - set(match_elements)
        # print('post',self.current_elements)

        return reward

    def _obs(self):
        g = nx.convert_node_labels_to_integers(self.current_graph)
        return self.embedding.embed(g, self.rng)/500

    def _record(self,d):
        if self.episode == 1 and self.time == 1:
            openchoice = 'w'
        else:
            openchoice = 'a'

        o = DictWriter(open(self.log_path, openchoice),['episode','time_tic',
            'current_elements',
            'current_nodes',
            'universe_elements',
            'universe_subsets',
            'susbet_size',
            'add_elements',
            'del_elements',
            'episode_rewards',
            'ave_rewards'
        ])

        if self.episode == 1 and self.time == 1:
            o.writeheader()

        o.writerow(d)


    def _reset(self):
        '''
        Clear the current pool and start a new one episode
        '''
        self.current_elements.clear()
        self.current_graph.clear()
        self.add_number = 0
        self.del_number = 0
        self.add_elements(self.arrive*3)
        self.elements_to_graph()
        self.time = 1
        self.reward = 0.0
        self.total_rewards = 0.0
        self.episode += 1 # start a new episode

        log_data = {
            'episode':self.episode,
            'time_tic':self.time,
            'current_elements':len(self.current_elements),
            'current_nodes':len(self.current_subsets),
            'universe_elements':self.N,
            'universe_subsets':self.l,
            'susbet_size':self.k,
            'add_elements':self.add_number,
            'del_elements':self.del_number,
            'episode_rewards':self.total_rewards/self.time,
            'ave_rewards': self.all_rewards / self.episode
        }

        if self.record == True:
            self._record(log_data)

        return self._obs()

    def _step(self,action):
        '''
        sequencial actions:
        1. match and return rewards
        2. vertices arrive and depart
        3. generate new observations and increase time
        '''
        if self.time < self.episode_length:
            if action == 1:
                reward = self.MIS_solver()
                self.total_rewards += reward
                self.all_rewards += reward
            elif action == 0:
                reward = 0
            else:
                print('illegal action!')
                raise NotImplementedError

            # generate new observation
            self.time += 1
            self.depart_elements()
            self.add_elements(self.arrive)
            self.elements_to_graph()

            if self.time == self.episode_length:
                self.done = True
            else:
                self.done = False

        else:
            reward = 0
            self.add_number = 0
            self.del_number = 0
            self.done = True

        log_data = {
            'episode': self.episode,
            'time_tic': self.time,
            'current_elements': len(self.current_elements),
            'current_nodes': len(self.current_subsets),
            'universe_elements': self.N,
            'universe_subsets': self.l,
            'susbet_size': self.k,
            'add_elements': self.add_number,
            'del_elements': self.del_number,
            'episode_rewards': self.total_rewards ,
            'ave_rewards': self.all_rewards / self.episode
        }
        print(self.record)
        if self.record == True:
            print('write')
            self._record(log_data)

        return self._obs(), reward, self.done, {}


