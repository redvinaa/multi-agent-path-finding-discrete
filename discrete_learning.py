# imports {{{
from common import DiscreteEnv, QLearningAgent, MultiAgent, LinearDecay, Net, BasicPolicy, Params, NOOPAgent
from tensorboardX import SummaryWriter
from multiprocessing import Process
import numpy as np
import torch.nn as nn
import torch
import argparse
import pickle
import os
import shutil
# }}}


def discrete_learning(args):
	if not 'ns' in args:
		args['ns'] = ''

	# initialization {{{ 
	dirname = os.path.dirname(os.path.abspath(__file__))
	dirname = os.path.join(dirname, 'runs/'+args["name"])
	if os.path.isdir(dirname):
		if args["delete_previous"]:
			shutil.rmtree(dirname, ignore_errors=True)
		else:
			print(f'{args["ns"]} > This run already exists')
			quit()
	params = Params[args["name"]]

	N_AGENTS = len(params['AGENTS'])

	# construct environment
	env = DiscreteEnv(N_AGENTS, params['MAP_IMAGE'])

	# load or construct Q_networks
	agent_list = []
	for i, name in enumerate(params['AGENTS']):
		if name == 'qlearning':
			net = Net(env.get_os_len(), 5, params['HIDDEN_LAYER_SIZE'], params['N_HIDDEN_LAYERS'])
			opt = torch.optim.Adam(net.parameters())
			agent = QLearningAgent(net, opt, None, params['GAMMA'])
		elif name == 'sarsa':
			net = Net(env.get_os_len(), 5, params['HIDDEN_LAYER_SIZE'], params['N_HIDDEN_LAYERS'])
			opt = torch.optim.Adam(net.parameters())
			agent = SARSAAgent(net, opt, None, params['GAMMA'])
		elif name == 'basic_policy':
			agent = BasicPolicy(env)
		elif name == 'noop':
			agent = NOOPAgent()
		else:
			raise Exception('No such agent')
		agent_list.append(agent)
	agents = MultiAgent(agent_list)

	if params['LOAD_MODEL']:
		agents = pickle.load(open(f'models/{params["LOAD_MODEL"]}.p', 'rb'))


	# log to tensorboard
	writer = SummaryWriter(f'runs/{args["name"]}')
	writer.add_text('parameters', str(params).replace('{', '').replace('}', '').replace(', ', '\n'))

	eps = LinearDecay(params['EPSILON_START'], params['EPSILON_FINAL'],
		params['EPSILON_DECAY_LENGTH'])
	steps = 0
	number_of_steps = np.zeros((N_AGENTS,))
	# }}}

	# training loop {{{
	S = env.reset()
	S = env.serialize(S)
	A = agents.reset(S)
	try:
		while True:
			if not args['quiet']:
				if steps % 1000 == 0:
					print(f'{args["ns"]} > Steps: {steps:.1e}')

			agents.epsilon = eps()
			writer.add_scalar('epsilon', eps(), steps)

			S, R, done, _ = env.step(A)
			S = env.serialize(S)
			A, L = agents.step(S, R, done)

			writer.add_scalar(f'action/agent_{env.curr_agent}', A, steps)
			if type(L) != type(None):
				writer.add_scalar(f'loss/agent_{env.curr_agent}', L, steps)

			# saving models
			if steps % (params['STEPS']//10) == 0 and steps != 0:
				print(f'{args["ns"]} > Saving models...')
				pickle.dump(agents, open(f'models/{args["name"]}.p', 'wb'))

			eps.step()
			steps += 1

			if done:
				writer.add_scalar(f'number_of_steps/agent_{env.curr_agent}',
					number_of_steps[env.curr_agent], steps)
				number_of_steps[env.curr_agent] = 0
			else:
				number_of_steps[env.curr_agent] += 1

			if steps > params['STEPS']:
				print(f'{args["ns"]} > Saving models...')
				pickle.dump(agents, open(f'models/{args["name"]}.p', 'wb'))
				break

			if args["render"]:
				env.render()

	except KeyboardInterrupt:
		print(f'{args["ns"]} > Saving models...')
		pickle.dump(agents, open(f'models/{args["name"]}.p', 'wb'))
	# }}}


if __name__ == '__main__':
	# parse arguments {{{
	parser = argparse.ArgumentParser()
	parser.add_argument('names', default='default', nargs='+', help='Names of the parameter groups')
	parser.add_argument('-r', '--render', action='store_true', help='Render environment')
	parser.add_argument('-q', '--quiet', action='store_true', help='Do not print progress')
	parser.add_argument('-d', '--delete_previous', action='store_true', help='If a run named "name" already exists, delete the old run')
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
			arg_dict.update({'name': name, 'ns': name})
			process = Process(target=discrete_learning, args=(arg_dict,))
			processes.append(process)
			process.start()

		try:
			for process in processes:
				process.join()
		except KeyboardInterrupt:
			pass
