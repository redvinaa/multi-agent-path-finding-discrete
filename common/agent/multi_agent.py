import numpy as np
import torch
import torch.nn as nn
from common.buffer import Experience

class MultiAgent: # {{{
	# this class can be used with an infinite episode multi-agent environment:
	# it wraps the agents, collects (X, A, R, X_) sets, then trains them

	def __init__(self, agent_list, epsilon=0, evolution_frequency=0):
		self.agent_list          = agent_list
		self.N                   = len(agent_list)
		self.epsilon             = epsilon
		self.evolution_frequency = evolution_frequency

		self.curr_agent = 0
		self.epoch_rewards = np.zeros((self.N,)) # rewards collected in one evolution step
		self.epoch_steps   = 0 # steps since last evolution
		self.reached_goal = []
		self.X  = []
		self.A  = []
		self.R  = []
		self.X_ = []

	def __getitem__(self, idx):
		return self.agent_list[idx]

	def reset(self, X):
		self.curr_agent = 0
		self.epoch_steps = 0
		self.epoch_rewards.fill(0)
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

	def __call__(self, X, idx=None):
		# action selector
		if idx == None:
			idx = self.curr_agent

		return self.agent_list[idx](X, self.epsilon)

	def evolution(self):
		best_agent = np.argmax(self.epoch_rewards)
		self.epoch_rewards.fill(0)
		best_agent_state_dict = self.agent_list[best_agent].net.state_dict()
		for n in range(self.N):
			if n == best_agent:
				continue

			self.agent_list[n].net.load_state_dict(best_agent_state_dict)

	def step(self, X, reward, reached_goal=False):
		# reward refers to the previous agent, state X refers to the current one
		X = X.copy()

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

			experience = Experience(
				self.X[self.curr_agent],
				self.A[self.curr_agent],
				self.R[self.curr_agent],
				self.reached_goal[self.curr_agent],
				self.X_[self.curr_agent])

			action, loss_val = self.agent_list[self.curr_agent].step(experience, self.epsilon)

			self.X[self.curr_agent] = X.copy()
			self.A[self.curr_agent] = action

		self.curr_agent = (self.curr_agent + 1) % self.N

		if self.evolution_frequency > 0:
			self.epoch_steps += 1
			self.epoch_rewards[self.curr_agent] += reward
			if self.curr_agent == 0 and self.epoch_steps >= self.evolution_frequency*self.N:
				self.evolution()

		return action, loss_val
# }}}

if __name__ == '__main__':
	from common.env import DiscreteEnv
	from common.agent import make_agent

	env = DiscreteEnv(1, 'maps/test_4x4.jpg')
	agent = make_agent('basic_policy', {'AGENTS':['basic_policy']})
	agents = MultiAgent([agent])

	S = env.reset()
	X = env.serialize(S)
	A = agents.reset(X)
	env.render()

	while True:
		S, R, info = env.step(A)
		X = env.serialize(S)
		A, _ = agents.step(X, 0)
		env.render()
