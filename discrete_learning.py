# imports {{{
from common.agent import SARSAAgent, MultiAgent
from common.base_policy import BasicPolicy
from common.env import DiscreteEnv
from common import Net, LinearDecay
from common.parameters import Params
from tensorboardX import SummaryWriter
import numpy as np
import torch.nn as nn
import torch
import argparse
import pickle
import os
import shutil
# }}}

# parse arguments {{{
parser = argparse.ArgumentParser()
parser.add_argument('name', default='default', nargs='?', help='Name of the parameter group')
parser.add_argument('-r', '--render', action='store_true', help='Render environment')
args = parser.parse_args()
# }}}

# initialization {{{ 
dirname = os.path.dirname(os.path.abspath(__file__))
dirname = os.path.join(dirname, 'runs/'+args.name)
if os.path.isdir(dirname):
	ans = input(f'This deletes directory \'{dirname}\'?\nAre you sure? ')
	assert(ans in ['y', 'yes', 'Y', 'n', 'N'])
	if ans in ['n', 'N']:
		quit()
	shutil.rmtree(dirname, ignore_errors=True)
params = Params[args.name]


# construct environment
env = DiscreteEnv(params['N_AGENTS'], params['MAP_IMAGE'])

# load or construct Q_network
net = Net(env.get_os_len(), 5, params['HIDDEN_LAYER_SIZE'], params['N_HIDDEN_LAYERS'])
opt = torch.optim.SGD(net.parameters(), lr=params['LR'], momentum=params['MOMENTUM'])
if params['STAGE'] == 1:
	policy = BasicPolicy(env)
else:
	policy = None
agent  = SARSAAgent(net, opt, policy, params['GAMMA'])
agents = MultiAgent(agent, params['N_AGENTS'])

if params['LOAD_MODEL']:
	agents = pickle.load(open(f'models/{params["LOAD_MODEL"]}.p', 'rb'))


# log to tensorboard
writer = SummaryWriter(f'runs/{args.name}')
writer.add_text('parameters', str(params).replace('{', '').replace('}', '').replace(', ', '\n'))

eps = LinearDecay(params['EPSILON_START'], params['EPSILON_FINAL'],
	params['EPSILON_DECAY_LENGTH'])
steps = 0
rewards = np.zeros((params['N_AGENTS'],))
number_of_steps = np.zeros((params['N_AGENTS'],))
# }}}


# training loop {{{
S = env.reset()
S = env.serialize(S)
A = agents.reset(S)
try:
	while True:
		if steps % 1000 == 0:
			print(f'Steps: {steps:.1e}')

		agents.epsilon = eps()
		writer.add_scalar('training/epsilon', eps(), steps)

		S, R, info = env.step(A)
		S = env.serialize(S)
		A, L = agents.step(S, R, info['reached_goal'])
		if type(L) != type(None):
			writer.add_scalar(f'training/loss', L, steps)

		rewards[env.curr_agent] += R
		if info['reached_goal']:
			writer.add_scalar(f'training/rewards', rewards[env.curr_agent], steps)
			rewards[env.curr_agent] = 0

		# saving models
		if params['MODEL_SAVE_FREQ'] > 0:
			if steps % params['MODEL_SAVE_FREQ'] == 0 and steps != 0:
				print('Saving models...')
				pickle.dump(agents, open(f'models/{args.name}.p', 'wb'))

		eps.step()
		steps += 1

		if info['reached_goal']:
			writer.add_scalar(f'training/number_of_steps',
				number_of_steps[agents.curr_agent], steps)
			number_of_steps[agents.curr_agent] = 0
		else:
			number_of_steps[agents.curr_agent] += 1

		if steps > params['STEPS']:
			print('Saving models...')
			pickle.dump(agents, open(f'models/{args.name}.p', 'wb'))
			break

		if args.render:
			env.render()

except KeyboardInterrupt:
	print('Saving models...')
	pickle.dump(agents, open(f'models/{args.name}.p', 'wb'))
# }}}
