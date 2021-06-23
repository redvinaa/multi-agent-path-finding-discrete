# Based on
# https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter06/02_dqn_pong.py

import numpy as np
import torch
from copy import deepcopy
import torch.nn as nn
from common.buffer import Experience, ExperienceBuffer

class DQNAgent:
	# this class includes the epsilon-greedy action-selector,
	# and the dqn algorithm (inputs: X, A, R, X_)
	# https://www.nature.com/articles/nature14236

	def __init__(self, net, optim,/,
		gamma=1., num_actions=5, sync_frequency=1, sample_size=1, buffer_size=1, device='cpu', policy=None):
		self.net            = net
		self.net_frozen     = deepcopy(net)
		self.optim          = optim
		self.gamma          = gamma
		self.num_actions    = num_actions
		self.sync_frequency = sync_frequency
		self.sample_size    = sample_size
		self.buffer_size    = buffer_size
		self.device         = device
		self.last_sync      = 0
		self.policy         = policy

		self.buffer = ExperienceBuffer(self.buffer_size)

	def __call__(self, X, epsilon=0.):
		# action selector

		if np.random.random() < epsilon:
			return np.random.randint(0, self.num_actions)

		if type(self.policy) == type(None):
			# greedy action from Q_network
			state_a = np.array([X], copy=False)
			state_v = torch.as_tensor(state_a)
			q_vals_v = self.net(state_v)
			_, act_v = torch.max(q_vals_v, dim=1)
			return int(act_v.flatten().item())

		# following policy
		return self.policy(X)

	def step(self, experience, epsilon=0.): # learning

		# store experience
		self.buffer.append(experience)

		# sample batch and learn
		loss_val = None
		if len(self.buffer) >= self.buffer_size:
			self.optim.zero_grad()

			batch = self.buffer.sample(self.sample_size)
			states, actions, rewards, dones, next_states = batch

			states_v      = torch.tensor(states).to(self.device)
			next_states_v = torch.tensor(next_states).to(self.device)
			actions_v     = torch.tensor(actions).to(self.device)
			rewards_v     = torch.tensor(rewards).to(self.device)
			done_mask     = torch.BoolTensor(dones).to(self.device)

			state_action_values = self.net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
			next_state_values = self.net_frozen(next_states_v).max(1)[0]
			next_state_values[done_mask] = 0.0
			next_state_values = next_state_values.detach()

			expected_state_action_values = next_state_values * self.gamma + rewards_v
			L = nn.MSELoss()(state_action_values, expected_state_action_values)

			loss_val = L.item()

			L.backward()
			self.optim.step()

		self.last_sync += 1
		if self.last_sync >= self.sync_frequency:
			self.net_frozen.load_state_dict(self.net.state_dict())

		X_ = experience.new_state
		A_  = self(X_, epsilon)
		return A_, loss_val
