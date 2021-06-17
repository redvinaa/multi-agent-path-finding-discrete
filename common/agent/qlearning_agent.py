import numpy as np
import torch
import torch.nn as nn
from common.buffer import Experience

class QLearningAgent:
	# this class includes the epsilon-greedy action-selector,
	# and the q-learning algorithm (inputs: X, A, R, X_)

	def __init__(self, net, optim, gamma=1., num_actions=5):
		self.net         = net
		self.optim       = optim
		self.gamma       = gamma
		self.num_actions = num_actions

	def __call__(self, X, epsilon=0.):
		# action selector

		if np.random.random() < epsilon:
			return np.random.randint(0, self.num_actions)

		state_a = np.array([X], copy=False)
		state_v = torch.as_tensor(state_a)
		q_vals_v = self.net(state_v)
		_, act_v = torch.max(q_vals_v, dim=1)
		return int(act_v.flatten().item())

	def step(self, experience, epsilon=0.): # learning
		self.optim.zero_grad()

		X, A, R, reached_goal, X_ = experience

		# transform to pytorch vectors
		Xv  = torch.tensor(X).unsqueeze(0)
		X_v = torch.tensor(X_).unsqueeze(0)
		A_  = self(X_, epsilon)

		Q_old = self.net(Xv)
		q_old = Q_old[0,A]

		if reached_goal:
			q_new = R
		else:
			Q_new = self.net(X_v).detach() # detach is very important
			q_new = R + self.gamma*torch.max(Q_new)

		q_new_ = torch.as_tensor(q_new)

		L = nn.MSELoss()( q_old, q_new_ )

		loss_val = L.item()

		L.backward()
		self.optim.step()

		Q_updated = self.net(Xv)

		return A_, loss_val
