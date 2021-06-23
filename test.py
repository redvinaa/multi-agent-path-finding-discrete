#!.venv/bin/python
# PYTHON_ARGCOMPLETE_OK

from common.env import DiscreteEnv
from common.agent import make_agent, MultiAgent
import torch
import numpy as np
import argparse, argcomplete, yaml, cv2


def test(args):

	with open('parameters.yaml', 'r') as f:
		Params = yaml.load(f, Loader=yaml.FullLoader)
	params = Params[args["name"]]

	# count agents
	N_AGENTS = 0
	for agent_obj in params['agents']:
		if type(agent_obj) == str:
			N_AGENTS += 1
		else:
			for i in range(agent_obj['n_agents']):
				N_AGENTS += 1

	# construct environment
	goal_closeness = None
	if 'goal_closeness' in params:
		# set goal closeness if needed
		goal_closeness = params['goal_closeness']
	env = DiscreteEnv(N_AGENTS, params['map_image'], params['obstacles'], goal_closeness)
	S = env.reset()
	print(f'{args["name"]} > Environment generated:\n'+
		f'\t- map: {params["map_image"]}\n'+
		f'\t- agents: {N_AGENTS}\n'+
		f'\t- obstacles: {params["obstacles"]}\n'+
		f'\t- observation_space_size: {env.get_os_len()}\n'+
		f'\t- max_distance: {np.max(env.cost_map)}')

	# construct agents
	agent_list = []
	for agent_obj in params['agents']:
		if type(agent_obj) == str: # only string is provided, no settings
			agent_type = agent_obj
			agent = make_agent(env, Params['agent_definitions'][agent_type])
			agent.net.load_state_dict(torch.load(f'models/{args["name"]}/{agent_type}.pt'))
			agent_list.append(agent)

		else: # settings are provided

			for i in range(agent_obj['n_agents']):

				agent = make_agent(env, Params['agent_definitions'][agent_obj['type']])
				agent.net.load_state_dict(torch.load(f'models/{args["name"]}/{agent_obj["type"]}.pt'))
				agent_list.append(agent)

	agents = MultiAgent(agent_list)
	agents.epsilon = args['epsilon']

	if args['video']:
		fourcc = cv2.VideoWriter_fourcc(*'MP4V')
		video = cv2.VideoWriter(args['video'][0], fourcc, 2*(N_AGENTS+params['obstacles']), (700, 700))

	reached_goal    = np.full((N_AGENTS, args['steps'],), False)


	try:
		# start playing episodes
		S = env.reset()
		for step in range(args['steps']):
			X = env.serialize(S)
			A = agents(X)

			if args['render']:
				env.render(500//N_AGENTS)
				#  env.render()

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
