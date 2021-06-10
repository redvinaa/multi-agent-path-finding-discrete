from common.env import pad_image
from common import DiscreteEnv, Params
from array2gif import write_gif
import numpy as np
import argparse
import pickle
import torch
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('name', default='default', nargs='?', help='Name of the parameters')
parser.add_argument('-r', '--render', action='store_true', help='Name of the parameters')
parser.add_argument('-s', '--steps', default=1000, type=int, help='How many episodes to run')
parser.add_argument('-e', '--epsilon', default=.0, type=float, help='Exploration factor')
args = parser.parse_args()
run_name = args.name
render   = args.render
steps    = args.steps
epsilon  = args.epsilon

params = Params[run_name]
N = len(params['AGENTS'])

# construct environment
env = DiscreteEnv(N, params['MAP_IMAGE'])

reached_goal    = np.full((N, steps,), False)

# load policy or Q_network
agents         = pickle.load(open(f"models/{run_name}.p", 'rb'))
agents.epsilon = epsilon


# start playing episodes
S = env.reset()
for step in range(steps):
	X = env.serialize(S)
	A = agents(X)

	if render:
		env.render(300//N)

	S, R, done, _ = env.step(A)
	reached_goal[env.curr_agent, step] = done

print(f'average steps to goal: {steps/np.sum(reached_goal, axis=1)}')
