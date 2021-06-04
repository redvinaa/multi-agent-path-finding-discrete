import numpy as np
import torch
import torch.nn as nn

class QLearningAgent: # {{{
	# this class includes the epsilon-greedy action-selector,
	# and the q-learning algorithm (inputs: X, A, R, X_)

	def __init__(self, net, optim, gamma=1):
		self.net   = net
		self.optim = optim
		self.gamma = gamma

	def greedy_action(self, X):
		state_a = np.array([X], copy=False)
		state_v = torch.tensor(state_a)
		q_vals_v = self.net(state_v)
		_, act_v = torch.max(q_vals_v, dim=1)
		return int(act_v.flatten().item())

	def __call__(self, X, epsilon=0):
		# action selector

		if np.random.random() < epsilon:
			return np.random.randint(0, 5)

		return self.greedy_action(X)

	def step(self, data, epsilon=0): # learning
		self.optim.zero_grad()

		X  = data['X']
		A  = data['A']
		R  = data['R']
		X_ = data['X_']
		reached_goal = data['reached_goal']

		Xv = torch.tensor(X).unsqueeze(0)
		X_v = torch.tensor(X_).unsqueeze(0)
		A_ = self(X_, epsilon)

		Q_old = self.net(Xv)[0,A]
		if reached_goal:
			Q_new = R
		else:
			Q_new = R + self.gamma*torch.max(self.net(X_v).detach())

		L = nn.MSELoss()( Q_new, Q_old )
		loss_val = L.item()

		L.backward()
		self.optim.step()

		return A_, loss_val
# }}}

class SARSAAgent: # {{{
	# this class includes the epsilon-greedy action-selector,
	# and the q-learning algorithm (inputs: X, A, R, X_)

	def __init__(self, net, optim, policy=None, gamma=1.):
		self.net    = net
		self.optim  = optim
		self.policy = policy
		self.gamma  = gamma

	def greedy_action(self, X):
		state_a = np.array([X], copy=False)
		state_v = torch.tensor(state_a)
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
		Xv = torch.tensor(X).unsqueeze(0)
		X_v = torch.tensor(X_).unsqueeze(0)
		A_ = self(X_, epsilon)

		Q_old = self.net(Xv)
		q_old = Q_old[0,A]

		Q_new = self.net(X_v).detach()
		if reached_goal:
			q_new = R
		else:
			q_new = R + self.gamma*Q_new[0, A_]

		Q_new[0, A_] = q_new

		L = nn.MSELoss()( q_new, q_old )

		loss_val = L.item()

		L.backward()
		self.optim.step()

		Q_updated = self.net(Xv)

		return A_, loss_val
# }}}

class MultiAgent: # {{{
	# this class can be used with an infinite episode multiagent environment:
	# it wraps the agents, collects (X, A, R, X_) sets, then trains them

	def __init__(self, agent, N, epsilon=0):
		self.agent   = agent
		self.N       = N
		self.epsilon = epsilon

		self.curr_agent = 0
		self.reached_goal = []
		self.X  = []
		self.A  = []
		self.R  = []
		self.X_ = []

	def reset(self, X):
		self.curr_agent = 0
		self.X.clear()
		self.A.clear()
		self.R.clear()
		self.X_.clear()
		self.reached_goal.clear()

		action = self(X)
		self.X.append(X.copy())
		self.A.append(action)

		self.curr_agent = (self.curr_agent + 1) % self.N
		return action

	def __call__(self, X):
		# action selector
		return self.agent(X, self.epsilon)

	def step(self, X, reward, reached_goal=False):
		# reward refers to the previous agent, state X refers to the current one

		if len(self.X) < self.N:
			# first round
			action = self(X)
			self.X.append(X.copy())
			self.R.append(reward)
			self.A.append(action)
			self.reached_goal.append(reached_goal)

			loss_val = None
		else:
			# not first round
			if len(self.X_) < self.N:
				self.X_.append(X.copy())
			else:
				self.X_[self.curr_agent] = X.copy()

			if len(self.R) < self.N:
				self.R.append(reward)
				self.reached_goal.append(reached_goal)
			else:
				self.R[(self.curr_agent-1)%self.N] = reward
				self.reached_goal[(self.curr_agent-1)%self.N] = reached_goal

			data = {
				'X':  self.X[self.curr_agent],
				'A':  self.A[self.curr_agent],
				'R':  self.R[self.curr_agent],
				'X_': self.X_[self.curr_agent],
				'reached_goal': self.reached_goal[self.curr_agent],
			}

			action, loss_val = self.agent.step(data, self.epsilon)

			self.X[self.curr_agent] = X.copy()
			self.A[self.curr_agent] = action

		self.curr_agent = (self.curr_agent + 1) % self.N
		return action, loss_val
# }}}

if __name__ == '__main__': # {{{
	from common.env import DiscreteEnv
	from common.base_policy import BasicPolicy
	from common import Net

	N = 1

	np.random.seed(0)
	env = DiscreteEnv(N, 'maps/empty_3x3.jpg')

	net = Net(env.get_os_len())
	opt = torch.optim.Adam(net.parameters())
	policy = BasicPolicy(env)
	policy = None
	agent = SARSAAgent(net, opt, policy, gamma=1.)
	agents = MultiAgent(agent, N, epsilon=.0)

	X = env.reset()
	env.render()
	X = env.serialize(X)
	A = agents.reset(X)

	while True:
		X, R, info = env.step(A)
		X = env.serialize(X)
		A, L = agents.step(X, R, info['reached_goal'])
		env.render()
# }}}
