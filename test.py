from common.env import DiscreteEnv
from array2gif import write_gif
import torch
import numpy as np
import argparse, pickle, yaml, cv2


def test(args):
	with open('parameters.yaml', 'r') as f:
		Params = yaml.load(f, Loader=yaml.FullLoader)
	params = Params[args['name']]
	N = sum(params['agents'].values())

	# construct environment
	env = DiscreteEnv(N, params['map_image'])

	reached_goal    = np.full((N, args['steps'],), False)

	# load policy or Q_network
	agents         = pickle.load(open(f"models/{args['name']}.p", 'rb'))
	agents.epsilon = args['epsilon']


	# start playing episodes
	S = env.reset()
	for step in range(args['steps']):
		X = env.serialize(S)
		A = agents(X)

		if args['render']:
			env.render(300//N)

		S, R, done, _ = env.step(A)
		reached_goal[env.curr_agent, step] = done

	print(f'average steps to goal: {args["steps"]/np.sum(reached_goal, axis=1)}')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('name', default='default', nargs='?', help='Name of the parameters')
	parser.add_argument('-r', '--render', action='store_true', help='Name of the parameters')
	parser.add_argument('-s', '--steps', default=1000, type=int, help='How many episodes to run')
	parser.add_argument('-e', '--epsilon', default=.0, type=float, help='Exploration factor')
	args = parser.parse_args()

	args_dict = vars(args)
	test(args_dict)
