import numpy as np
import torch
import torch.nn as nn

from common.agent.qlearning_agent import QLearningAgent
from common.agent.dqn_agent import DQNAgent
from common.agent.multi_agent import MultiAgent
from common.agent.net import Net
from common.agent.basic_policy import BasicPolicy

def make_agent(env, conf):
	net = Net(
		env.get_os_len(),
		env.action_space.n,
		conf['hidden_layer_size'],
		conf['n_hidden_layers'])
	opt = torch.optim.Adam(net.parameters())

	if conf['type'] == 'qlearning':
		return QLearningAgent(
			net,
			opt,
			conf['gamma'],
			env.action_space.n)

	if conf['type'] == 'DQN':
		return DQNAgent(net,
			opt,
			conf['gamma'],
			env.action_space.n,
			conf['sync_frequency'],
			conf['sample_size'],
			conf['buffer_size'],
			conf['device'])

	raise ValueError(f'No such agent: {conf["type"]}')
