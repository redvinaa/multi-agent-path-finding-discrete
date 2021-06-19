#!.venv/bin/python
# PYTHON_ARGCOMPLETE_OK

# imports {{{
from common.linear_decay import LinearDecay
from common.agent import make_agent, MultiAgent
from common.env import DiscreteEnv
from tensorboardX import SummaryWriter
from multiprocessing import Process
import torch.nn as nn
import torch
import numpy as np
import argparse, argcomplete, pickle, shutil, yaml, os, time
# }}}


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

	N_AGENTS = sum(params['agents'].values())

	# construct environment
	env = DiscreteEnv(N_AGENTS, params['map_image'], params['obstacles'], 1)
	print(f'{args["name"]} > Environment generated:\n'+
		f'\t- map: {params["map_image"]}\n'+
		f'\t- agents: {N_AGENTS}\n'+
		f'\t- obstacles: {params["obstacles"]}\n'+
		f'\t- observation_space_size: {env.get_os_len()}\n'+
		f'\t- max_distance: {np.max(env.cost_map)}')

	# load or construct Q_networks
	agent_list = []
	for agent_type, num in params['agents'].items():
		for i in range(num):
			agent = make_agent(env, Params['agent_definitions'][agent_type])
			agent_list.append(agent)
	agents = MultiAgent(agent_list)

	if params['load_model']:
		agents = pickle.load(open(f'models/{params["load_model"]}.p', 'rb'))


	# log to tensorboard
	writer = SummaryWriter(f'runs/{args["name"]}')
	writer.add_text('parameters', str(params).replace('{', '').replace('}', '').replace(', ', '\n'))

	eps = LinearDecay(params['epsilon_start'], params['epsilon_final'],
		params['epsilon_decay_length'])
	goal_cl = LinearDecay(params['goal_closeness_start'], params['goal_closeness_final'],
		params['goal_closeness_decay_length'])
	steps = 0
	number_of_steps = np.zeros((N_AGENTS,))
	# }}}

	# training loop {{{
	agents.epsilon = eps()
	env.goal_closeness = goal_cl()

	episode_start_time = time.time()

	S = env.reset()
	S = env.serialize(S)
	A = agents.reset(S)
	try:
		while True:
			if not args['quiet']:
				if steps % 1000 == 0:
					print(f'{args["name"]} > Steps: {steps:.1e}')

			agents.epsilon = eps()
			env.goal_closeness = goal_cl()
			writer.add_scalar('global/epsilon', eps(), steps)
			writer.add_scalar('global/goal_closeness', goal_cl(), steps)

			S, R, done, _ = env.step(A)
			S = env.serialize(S)
			A, L = agents.step(S, R, done)

			if type(L) != type(None):
				writer.add_scalar(f'loss/agent_{env.curr_agent}', L, steps)

			# saving models
			if steps % (params['steps']//10) == 0 and steps != 0:
				print(f'{args["name"]} > Saving models...')
				pickle.dump(agents, open(f'models/{args["name"]}.p', 'wb'))

			eps.step()
			goal_cl.step()
			steps += 1

			if done:
				writer.add_scalar(f'number_of_steps/agent_{env.curr_agent}',
					number_of_steps[env.curr_agent], steps)
				number_of_steps[env.curr_agent] = 0

				writer.add_scalar('global/episode_time', time.time() - episode_start_time, steps)
				episode_start_time = time.time()
			else:
				number_of_steps[env.curr_agent] += 1

			if steps > params['steps']:
				print(f'{args["name"]} > Saving models...')
				pickle.dump(agents, open(f'models/{args["name"]}.p', 'wb'))
				break

			if args["render"]:
				env.render()

	except KeyboardInterrupt:
		print(f'{args["name"]} > Saving models...')
		pickle.dump(agents, open(f'models/{args["name"]}.p', 'wb'))
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
