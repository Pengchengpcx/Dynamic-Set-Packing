from mpi4py import MPI
from baselines.common import set_global_seeds
import os.path as osp
import os
import logging
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.common.mpi_fork import mpi_fork
from baselines import bench
from baselines.trpo_mpi import trpo_mpi
import trpo_indi
import sys
import gym
import set_packing
from set_packing import wrapper, embeddings




# Training parameters

# Note: incorporate the k to the equation
max_kl=0.01
cg_iters=10
cd_damping=0.1
gamma=0.99
lam=0.98
vf_iters=5
vf_stepsize=1e-6
num_cpu=1



def train(env_id, num_timesteps, seed, model_path, load_model,
          timesteps_per_batch,hidden_units,hidden_layers, para, EMBEDDING):
    whoami  = mpi_fork(num_cpu)
    if whoami == "parent":
        return
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)

    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = gym.make(env_id)
    env = wrapper.SetpackingWrapper(env, para, EMBEDDING)
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name,
                         ob_space=env.observation_space,
                         ac_space=env.action_space,
                         hid_size=hidden_units,
                         num_hid_layers=hidden_layers)
    env.seed(workerseed)

    trpo_indi.learn(env, policy_fn,
                   timesteps_per_batch=timesteps_per_batch,
                   max_kl=max_kl, cg_iters=cg_iters,
                   cg_damping=cd_damping,
                   max_episodes=num_timesteps,
                   gamma=gamma, lam=lam,
                   vf_iters=vf_iters,
                   vf_stepsize=vf_stepsize,
                   load_model=load_model,
                   model_path=model_path
                    )
    env.close()

def main():
    '''
    All Input Parameters
    '''
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--loadmodel', help='Load the Neural Net', type=bool, default=False)
    parser.add_argument('--trainsize', help='Training trajecotries', type=int, default=8000)
    parser.add_argument('--batchsize', help='Batch time steps in each update', type=int, default=800)
    parser.add_argument('--hiddenunit', help='Hidden units for each layer in Neural Net', type=int, default=150)
    parser.add_argument('--hiddenlayers', help='Hidden layers for each layer in Neural Net', type=int, default=4)
    parser.add_argument('--N', help='total vertices', type=int, default=300)
    parser.add_argument('--k', help='subset size', type=int, default=3)
    parser.add_argument('--l', help='total subsets', type=int, default=500)
    parser.add_argument('--length', help='episode length', type=int, default=500)
    parser.add_argument('--arrive', help='arrive rate', type=int, default=50)
    parser.add_argument('--depart', help='depart rate', type=int, default=40)
    parser.add_argument('--record', help='record process', type=bool, default=True)
    parser.add_argument('--load', help='load data', type=bool, default=True)





    args = parser.parse_args()

    '''
    Environment Parameters Setting
    '''
    env_id = "Dynamic_setpacking-v0"

    '''
    Training Parameters
    '''
    maxepisodes = args.trainsize
    timesteps_per_batch = args.batchsize
    hidden_units = args.hiddenunit
    hidden_layers = args.hiddenlayers
    N = args.N
    k = args.k
    l = args.l
    arrive = args.arrive
    depart = args.depart
    length = args.length
    record = args.record
    load = args.load

    TAU = 5
    ALPHA = 0.05
    EMBEDDING = embeddings.Walk2VecEmbedding([embeddings.p0_max, embeddings.p0_mean], TAU, ALPHA)
    para = {
        'N': N,
        'k': k,
        'l': l,
        'arrive': arrive,
        'depart': depart,
        'episode_length': length,
        'record':record,
        'load':load,
        'agent':'trpo'
    }


    model_path = './NN/N%sk%sl%sarrive%sdepart%s/' % (N, k, l, arrive, depart)
    load_model = args.loadmodel


    train(env_id=env_id, num_timesteps=maxepisodes, seed=0,  model_path=model_path,
          load_model=load_model,timesteps_per_batch=timesteps_per_batch,
          hidden_layers=hidden_layers, hidden_units=hidden_units, para=para, EMBEDDING= EMBEDDING
          )

if __name__ == '__main__':
    main()