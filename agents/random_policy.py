import gym
import set_packing
from set_packing import wrapper, embeddings
import numpy as np

TAU = 5
ALPHA = 0.05
SHOW = False
EMBEDDING = embeddings.Walk2VecEmbedding([embeddings.p0_max, embeddings.p0_mean],TAU,ALPHA)

# MAIN
def main():
	import argparse
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--N', help='total vertices', type=int, default=300)
	parser.add_argument('--k', help='subset size', type=int, default=3)
	parser.add_argument('--l', help='total subsets', type=int, default=500)
	parser.add_argument('--length', help='episode length', type=int, default=500)
	parser.add_argument('--arrive', help='arrive rate', type=int, default=50)
	parser.add_argument('--depart', help='depart rate', type=int, default=40)
	parser.add_argument('--record', help='record process', type=bool, default=True)
	parser.add_argument('--load', help='load data', type=bool, default=True)
	parser.add_argument('--epi', help='episodes', type=int, default=800)

	args = parser.parse_args()

	N = args.N
	k = args.k
	l = args.l
	arrive = args.arrive
	depart = args.depart
	length = args.length
	record = args.record
	load = args.load
	EPISODES = args.epi

	para = {
		'N': N,
		'k': k,
		'l': l,
		'arrive': arrive,
		'depart': depart,
		'episode_length': length,
		'record': record,
		'load': load,
		'agent': 'random'
	}

	env = gym.make("Dynamic_setpacking-v0")
	env = wrapper.SetpackingWrapper(env, para, EMBEDDING )

	for i in range(EPISODES):
		obs, done = env.reset(), False
		while not done:
			a = np.random.randint(0,2)
			obs, reward, done, _ = env.step(a)

if __name__ == "__main__":
	main()