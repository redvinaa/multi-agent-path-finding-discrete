import numpy as np
import torch
import torch.nn as nn

class SARSAAgent:
	# this class includes the epsilon-greedy action-selector,
	# and the sarsa algorithm (inputs: X, A, R, X_)

	def __init__(self, net, optim, policy=None, gamma=1.):
		raise NotImplementedError('This is not updated so it surely does not work')
		self.net    = net
		self.optim  = optim
		self.policy = policy
		self.gamma  = gamma

	def greedy_action(self, X):
		state_a = np.array([X], copy=False)
		state_v = torch.as_tensor(state_a)
		q_vals_v = self.net(state_v)
		_, act_v = torch.max(q_vals_v, dim=1)
		return int(act_v.flatten().item())

	def __call__(self, X, epsilon=0.):
		# action selector

		if np.random.random() < epsilon:
			return np.random.randint(0, 5)

		if self.policy == None:
			return self.greedy_action(X)
		return self.policy(X)

	def step(self, data, epsilon=0.): # learning
		self.optim.zero_grad()

		X  = data['X']
		A  = data['A']
		R  = data['R']
		X_ = data['X_']
		reached_goal = data['reached_goal']

		# transform to pytorch vectors
		Xv = torch.as_tensor(X).unsqueeze(0)
		X_v = torch.as_tensor(X_).unsqueeze(0)
		A_ = self(X_, epsilon)

		Q_old = self.net(Xv)
		q_old = Q_old[0,A]

		if reached_goal:
			q_new = R
		else:
			Q_new = self.net(X_v).detach() # detach is very important
			q_new = R + self.gamma*Q_new[0, A_]

		q_new_ = torch.as_tensor(q_new)

		L = nn.MSELoss()( q_new_, q_old )

		loss_val = L.item()

		L.backward()
		self.optim.step()

		Q_updated = self.net(Xv)

		return A_, loss_val
