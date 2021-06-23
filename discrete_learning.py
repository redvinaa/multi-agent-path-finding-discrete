#!.venv/bin/python
# PYTHON_ARGCOMPLETE_OK

# imports {{{
from common.linear_decay import LinearDecay
from common.agent import make_agent, MultiAgent
from common.policy import make_policy
from common.env import DiscreteEnv
from tensorboardX import SummaryWriter
from multiprocessing import Process
import torch.nn as nn
import torch
import numpy as np
import argparse, argcomplete, shutil, yaml, os, time
# }}}


def save_models(run_name, agent_list, unique_agent_indices):
	# loop through agent types
	# save only one of each, agents are similar because of the evolution anyway
	for agent_type, idx in unique_agent_indices.items():
		torch.save(agent_list[idx].net.state_dict(), f'models/{run_name}/{agent_type}.pt')

def discrete_learning(args):
	# initialization {{{ 
	dirname = os.path.dirname(os.path.abspath(__file__))
	dirname = os.path.join(dirname, 'runs/'+args["name"])
	if os.path.isdir(dirname):
		if args["delete_previous"]:
			shutil.rmtree(dirname, ignore_errors=True)
		else:
			print(f'{args["name"]} > This run already exists')
			quit()

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

	# check if a folder exists for the models to be saved in, else create it
	if not os.path.isdir(f'models/{args["name"]}'):
		os.mkdir(f'models/{args["name"]}')

	# construct agents
	unique_agent_indices = {} # store the index of 1 agent from each type (for saving models)
	idx = 0
	agent_list = []
	for agent_obj in params['agents']:
		if type(agent_obj) == str: # only string is provided, no settings
			agent_type = agent_obj
			agent = make_agent(env, Params['agent_definitions'][agent_type])
			agent_list.append(agent)

			unique_agent_indices.update({agent_type: idx})
			idx += 1

		else: # settings are provided

			for i in range(agent_obj['n_agents']):

				policy = None
				if 'policy' in agent_obj:
					policy = make_policy(env, agent_obj['policy'])

				agent = make_agent(env, Params['agent_definitions'][agent_obj['type']], policy)

				if 'load' in agent_obj:
					agent.net.load_state_dict(torch.load(f'models/{agent_obj["load"]}.pt'))
				agent_list.append(agent)

				unique_agent_indices.update({agent_obj['type']: idx})
				idx += 1

	agents = MultiAgent(agent_list, evolution_frequency=params['evolution_frequency'])

	# log to tensorboard
	writer = SummaryWriter(f'runs/{args["name"]}')
	writer.add_text('parameters', str(params).replace('{', '').replace('}', '').replace(', ', '\n'))

	eps = LinearDecay(params['epsilon_start'], params['epsilon_final'],
		params['epsilon_decay_length']*N_AGENTS)

	steps = 0
	number_of_steps = np.zeros((N_AGENTS,))
	# }}}

	# training loop {{{
	agents.epsilon = eps()

	episode_start_time = time.time()

	#  S = env.reset() # this was called before, to get the stats
	S = env.serialize(S)
	A = agents.reset(S)
	try:
		while True:
			if not args['quiet']:
				if steps % 1000 == 0:
					print(f'{args["name"]} > Steps: {steps:.1e}')

			for n in range(N_AGENTS):

				agents.epsilon = eps()
				writer.add_scalar('global/epsilon', eps(), steps)

				S, R, done, _ = env.step(A)
				S = env.serialize(S)
				A, L = agents.step(S, R, done)

				if type(L) != type(None):
					writer.add_scalar(f'loss/agent_{env.curr_agent}', L, steps)

				if done:
					writer.add_scalar(f'number_of_steps/agent_{env.curr_agent}',
						number_of_steps[env.curr_agent], steps)
					number_of_steps[env.curr_agent] = 0

					writer.add_scalar('global/episode_time', time.time() - episode_start_time, steps)
					episode_start_time = time.time()
				else:
					number_of_steps[env.curr_agent] += 1

				if args["render"]:
					env.render()

			# saving models
			if steps % (params['steps']//10) == 0 and steps != 0:
				print(f'{args["name"]} > Saving models...')
				save_models(args['name'], agents.agent_list, unique_agent_indices)

			if steps > params['steps']:
				break

			eps.step()
			steps += 1

	except KeyboardInterrupt:
		pass

	finally:
		print(f'{args["name"]} > Saving models...')
		save_models(args['name'], agents.agent_list, unique_agent_indices)
	# }}}


if __name__ == '__main__':
	# get available configurations
	with open('parameters.yaml', 'r') as f:
		Params = yaml.load(f, Loader=yaml.FullLoader)
	choices = np.array(list(Params.keys()))
	choices = choices[choices != 'agent_definitions']

	# parse arguments {{{
	parser = argparse.ArgumentParser()
	parser.add_argument('names', default='default', nargs='+', choices=choices, help='Names of the parameter groups')
	parser.add_argument('-r', '--render', action='store_true', help='Render environment')
	parser.add_argument('-q', '--quiet', action='store_true', help='Do not print progress')
	parser.add_argument('-d', '--delete_previous', action='store_true', help='If a run named "name" already exists, delete the old run')
	argcomplete.autocomplete(parser)
	args = parser.parse_args()
	# }}}

	if len(args.names) == 1:
		# dont do multiprocessing
		args.name = args.names[0]
		arg_dict = vars(args)
		discrete_learning(arg_dict)

	else:
		# start learning in separate processes
		arg_dict = vars(args)
		processes = []
		for name in args.names:
			arg_dict.update({'name': name})
			process = Process(target=discrete_learning, args=(arg_dict,))
			processes.append(process)
			process.start()

		try:
			for process in processes:
				process.join()
		except KeyboardInterrupt:
			pass
