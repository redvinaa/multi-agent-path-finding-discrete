import numpy as np
#  np.seterr(all='ignore') # suppress nan warning

class BasicPolicy:
	def __init__(self, env):
		self.env = env
		self.ACTION_DELTAS = np.array([[0, 1], [-1, 0], [0, -1], [1, 0], [0, 0]])

	def __call__(self, X):
		S = self.env.deserialize(X, int)
		local_map = S[0]
		cost_map  = S[2][0] # in the list of cost_maps, the first one refers to the current agent

		lens = np.empty((5,))
		for a in range(5):
			i1 = self.ACTION_DELTAS[a, 0] + 1
			i2 = self.ACTION_DELTAS[a, 1] + 1
			l = cost_map[i1, i2]
			if l == -1 or local_map[i1, i2] == 0:
				lens[a] = np.nan
			else:
				lens[a] = l

		try:
			return np.nanargmin(lens)
		except ValueError:
			# no valid steps, chose random
			return np.random.randint(0, self.env.action_space.n)
