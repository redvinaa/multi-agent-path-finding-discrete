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
