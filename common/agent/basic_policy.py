import numpy as np

class BasicPolicy:
	def __init__(self, env):
		raise NotImplementedError('This is not updated so it surely does not work')

		# env is needed to deserialize observations
		self.env = env
		self.ACTION_DELTAS = np.array([[0, 0], [0, 1], [-1, 0], [0, -1], [1, 0]])

	def __call__(self, X, epsilon=0):
		# epsilon is only here for MultiAgent compatibility
		S = env.deserialize(X, int)
		cost_map_self = S[2][0]
		lens = np.empty((5,))
		for a in range(5):
			i1 = self.ACTION_DELTAS[a, 0] + 1
			i2 = self.ACTION_DELTAS[a, 1] + 1
			l = cost_map_self[i1, i2]
			if l == -1:
				lens[a] = np.nan
			else:
				lens[a] = l
		return np.nanargmin(lens)

	def step(self, data, epsilon=0):
		# this is a dummy function to that the policy fits into
		# the MultiAgent class

		return self(data['X_']), None

if __name__ == '__main__':
	from common.env import DiscreteEnv
	env = DiscreteEnv(1, 'maps/test_4x4.jpg')
	agent = BasicPolicy(1)

	S = env.reset()
	X = env.serialize(S)
	env.render()
	A = agent(X)
	done = False

	while not done:
		S, R, info = env.step(A)
		X = env.serialize(S)
		env.render()
		A = agent(X)
