import numpy as np

class BasicPolicy: # {{{
	def __init__(self, env):
		self.env = env
		self.ACTION_DELTAS = np.array([[0, 0], [0, 1], [-1, 0], [0, -1], [1, 0]])

	def __call__(self, X):
		S = self.env.deserialize(X, int)
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
# }}}

class DMRCPPolicy: # {{{ TODO this is not used

	# based on the following paper, although not much of it is used
	#   @article{DMRCP,
	#   author = {Wei, Changyun and Hindriks, Koen and Jonker, Catholijn},
	#   year = {2014},
	#   month = {06},
	#   pages = {21-31},
	#   title = {Multi-robot Cooperative Pathfinding: A Decentralized Approach},
	#   volume = {8481},
	#   journal = {Lecture Notes in Artificial Intelligence (Subseries of Lecture Notes in Computer Science)},
	#   doi = {10.1007/978-3-319-07455-9_3}
	#   }

	def __init__(self, N, lidar_radius=1, broadcast_path_length=2): # {{{
		self.N                     = N
		self.lidar_radius          = lidar_radius
		self.broadcast_path_length = broadcast_path_length
		assert(self.broadcast_path_length >= 2)

		self.ACTION_DELTAS = np.array([[0, 0], [0, 1], [-1, 0], [0, -1], [1, 0]])
		raise Exception('Don\'t use this yet.')
	# }}}

	def get_action(self, next_state): # {{{
		return np.flatnonzero((self.ACTION_DELTAS==next_state).all(1))[0]
	# }}}

	def __call__(self, observation): # {{{
		observation = observation.copy()

		## deserialize observation
		# get loc_map from observation
		loc_map_size = self.lidar_radius*2+1
		loc_map = observation[:loc_map_size**2].reshape(loc_map_size, loc_map_size)
		observation = observation[loc_map_size**2:]
		if self.lidar_radius != 1:
			# get inner 3x3 pixels
			loc_map = loc_map[self.lidar_radius-1:self.lidar_radius+2, self.lidar_radius-1:self.lidar_radius+2]

		# get global plan of current robot
		plan_self = observation[:self.broadcast_path_length*2].reshape(self.broadcast_path_length, 2)
		observation = observation[self.broadcast_path_length*2:]

		# get path_len
		path_len = observation[:1]
		observation = observation[1:]

		# get coords of agents
		states_other = observation[:(self.N-1)*2].reshape(self.N-1, 2)
		observation = observation[(self.N-1)*2:]

		# get global_plan
		plan_other = observation.reshape(self.N-1, self.broadcast_path_length, 2)

		## algorithm
		V   = np.array([0, 0])  # current state
		U_1 = plan_self[0] # broadcast path
		U_2 = plan_self[1] # broadcast path

		if (V == U_1).all():
			# already at goal
			return 0

		# planned next states of other agents
		next_states  = plan_other[:, 0]
		next_states_ = plan_other[:, 1] # next next

		if len(np.flatnonzero((states_other==U_1).all(1))):
			# another agent is where we want to go

			return 0

			# this would work, but let's make the RL algorithm solve this
			s_idx = np.flatnonzero((states_other==U_1).all(1))[0] # where next_states == U_1
			if (V != next_states[s_idx]).all():
				# don't need to switch places
				return 0
			else:
				# need to switch places

				# get list of free neighbouring cells
				F = []
				for a in range(1, 5):
					delta = self.ACTION_DELTAS[a]
					if loc_map[self.lidar_radius+delta[0], self.lidar_radius+delta[1]] == 1:
						F.append(V + delta)

				for K in F:
					if not (K == next_states_[s_idx]).all():
						# dodging
						return self.get_action(K - V)

				s_rel_loc = states_other[s_idx] - V + np.array([self.lidar_radius, self.lidar_radius])
				# check if s can dodge
				for a in range(1, 5): # actions s can take
					 if a == self.get_action(V - states_other[s_idx]):
						 # V is marked free on the local map, but its not a viable step for s
						 # because agent r is there
						 continue
					 delta = self.ACTION_DELTAS[a]
					 K = s_rel_loc + delta
					 if self.lidar_radius*2+1 in K or -1 in K:
						 # we can't see these
						 continue
					 K = K.astype(int)
					 if loc_map[K[0], K[1]] == 1:
						 # exists K, waiting
						 return 0

				if len(F) > 0:
					# retreating
					return self.get_action(F[0] - V)

				# waiting
				return 0

		else:
			return self.get_action(U_1 - V)
	# }}}
# }}}

if __name__ == '__main__':
	from common.env import DiscreteEnv

	np.random.seed(0)
	N = 2

	env = DiscreteEnv(N, 'maps/test.jpg')
	policy = BasicPolicy(env)


	S = env.reset()

	for _ in range(20*N):
		env.render()
		X = env.serialize(S)
		A = policy(X)
		S, R, info = env.step(A)

	env.render()
