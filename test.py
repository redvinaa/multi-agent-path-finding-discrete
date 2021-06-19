#!.venv/bin/python
# PYTHON_ARGCOMPLETE_OK

from common.env import DiscreteEnv
import torch
import numpy as np
import argparse, argcomplete, pickle, yaml, cv2


def test(args):
	with open('parameters.yaml', 'r') as f:
		Params = yaml.load(f, Loader=yaml.FullLoader)
	params = Params[args['name']]
	N = sum(params['agents'].values())

	if args['video']:
		fourcc = cv2.VideoWriter_fourcc(*'MP4V')
		video = cv2.VideoWriter(args['video'][0], fourcc, 1/(500//2/1000), (700, 700))

	# construct environment
	env = DiscreteEnv(N, params['map_image'])

	reached_goal    = np.full((N, args['steps'],), False)

	# load policy or Q_network
	agents         = pickle.load(open(f"models/{args['name']}.p", 'rb'))
	agents.epsilon = args['epsilon']


	try:
		# start playing episodes
		S = env.reset()
		for step in range(args['steps']):
			X = env.serialize(S)
			A = agents(X)

			if args['render']:
				env.render(500//N)

			if args['video']:
				frame = env.get_rendered_pic()
				frame = (frame * 255).astype(np.uint8)
				video.write(frame)

			S, R, done, _ = env.step(A)
			reached_goal[env.curr_agent, step] = done

		print(f'average steps to goal: {args["steps"]/np.sum(reached_goal, axis=1)}')
	except KeyboardInterrupt:
		pass

	if args['video']:
		print(f'Writing {args["video"][0]}')
		video.release()
		print('Done!')

if __name__ == '__main__':
	# get available configurations
	with open('parameters.yaml', 'r') as f:
		Params = yaml.load(f, Loader=yaml.FullLoader)
	choices = np.array(list(Params.keys()))
	choices = choices[choices != 'agent_definitions']

	# parse arguments {{{
	parser = argparse.ArgumentParser()
	parser.add_argument('name', default='default', nargs='?', choices=choices, help='Name of the parameters')
	parser.add_argument('-r', '--render', action='store_true', help='Name of the parameters')
	parser.add_argument('-s', '--steps', default=1000, type=int, help='How many episodes to run')
	parser.add_argument('-e', '--epsilon', default=.0, type=float, help='Exploration factor')
	parser.add_argument('--video', nargs=1, type=str, help='Save video with given name (.mp4 format)')
	argcomplete.autocomplete(parser)
	args = parser.parse_args()
	# }}}

	args_dict = vars(args)
	test(args_dict)
