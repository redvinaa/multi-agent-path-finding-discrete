from common.env import DiscreteEnv, pad_image
from common.parameters import Params
from array2gif import write_gif
import torch
import numpy as np
import argparse
import pickle
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('name', default='default', nargs='?', help='Name of the parameters')
parser.add_argument('-r', '--render', action='store_true', help='Name of the parameters')
parser.add_argument('-s', '--steps', default=1000, type=int, help='How many episodes to run')
parser.add_argument('-e', '--epsilon', default=.1, type=float, help='Exploration factor')
args = parser.parse_args()
run_name = args.name
render   = args.render
steps    = args.steps
epsilon  = args.epsilon

params = Params[run_name]

# construct environment
env = DiscreteEnv(params['N_AGENTS'], params['MAP_IMAGE'])
os_len = env.get_os_len()
print(f'Size of the state vector: {os_len}')

rewards      = np.zeros((params['N_AGENTS'],))
reached_goal = np.full((params['N_AGENTS'], steps,), False)

# load policy or Q_network
agents = pickle.load(open(f"models/{run_name}.p", 'rb'))
agents.epsilon = epsilon


# start playing episodes
S = env.reset()
for step in range(steps):
	X = env.serialize(S)
	A = agents(X)

	if render:
		env.render(300//params['N_AGENTS'])

	S, R, info = env.step(A)

	reached_goal[env.curr_agent, step] = info['reached_goal']
	rewards[env.curr_agent] += R

print(f'average rewards: {rewards/steps}')
print(f'average steps to goal: {steps/np.sum(reached_goal, axis=1)}')
