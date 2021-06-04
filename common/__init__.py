import torch
import torch.nn as nn
import numpy as np


# neural network class
class Net(nn.Module):
	def __init__(self, input_size, n_actions=5, hidden_layer_size=10, n_hidden_layers=1):
		super(Net, self).__init__()

		self.hidden = nn.ModuleList()

		self.hidden.append(nn.Linear(input_size, hidden_layer_size))
		self.hidden.append(nn.ReLU())

		for k in range(n_hidden_layers-1):
			self.hidden.append(nn.Linear(hidden_layer_size, hidden_layer_size))
		self.hidden.append(nn.Linear(hidden_layer_size, n_actions))

	def forward(self, x):
		for h_lay in self.hidden:
			x = h_lay(x)

		return x


def EGreedyActionSelector(state, net, epsilon=0):
	if np.random.random() < epsilon:
		return np.random.randint(0, 5)
	else:
		state_a = np.array([state], copy=False)
		state_v = torch.tensor(state_a)
		q_vals_v = net(state_v)
		_, act_v = torch.max(q_vals_v, dim=1)
		action = int(act_v.flatten().item())
		return action


# linear decay for epsilon
class LinearDecay:
	def __init__(self, start, end, max_steps):
		self.start = start
		self.end = end
		self.max_steps = max_steps
		self.curr_steps = 0

		self.val = start
		if max_steps > 0:
			self.delta = (start - end) / max_steps

	def step(self):
		if self.max_steps > 0:
			if self.start > self.end:
				self.val = max(self.val - self.delta, self.end)
			else:
				self.val = min(self.val - self.delta, self.end)

	def __call__(self):
		# if max_steps <= 0, self.val is always = start
		return self.val
