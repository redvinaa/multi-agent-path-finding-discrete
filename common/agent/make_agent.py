from common.agent.qlearning_agent import QLearningAgent
from common.agent.dqn_agent import DQNAgent
from common.agent.multi_agent import MultiAgent
from common.agent.net import Net
import torch

def make_agent(env, conf, policy=None):
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
			gamma=conf['gamma'],
			num_actions=env.action_space.n,
			sync_frequency=conf['sync_frequency'],
			sample_size=conf['sample_size'],
			buffer_size=conf['buffer_size'],
			device=conf['device'],
			policy=policy)

	raise ValueError(f'No such agent: {conf["type"]}')
