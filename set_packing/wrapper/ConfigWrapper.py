import gym
from gym import Wrapper
from csv import DictWriter

class SetpackingWrapper(Wrapper):
    def __init__(self,env,para,embedding):
        env = env.unwrapped

        if 'data_path' in para: env.path = para.pop('data_path')
        if 'log_path' in para: env.log_path = para.pop('log_path')
        if 'N' in para: env.N = para.pop('N')
        if 'k' in para: env.k = para.pop('k')
        if 'l' in para: env.l = para.pop('l')
        if 'arrive' in para: env.arrive = para.pop('arrive')
        if 'depart' in para: env.depart = para.pop('depart')
        if 'episode_length' in para: env.episode_length = para.pop('episode_length')
        if 'record' in para: env.record = para.pop('record')
        if 'load' in para: env.load = para.pop('load')
        if 'agent' in para: env.agent = para.pop('agent')



        env.embedding = embedding
        embedding.env = env
        env.setup()

        super(SetpackingWrapper, self).__init__(env)



