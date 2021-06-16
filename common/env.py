from gym import spaces
import numpy as np
import gym
import cv2
import os
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

def pad_image(image, size=1): # {{{
	return cv2.copyMakeBorder(image, size, size, size, size, cv2.BORDER_CONSTANT)
# }}}

class GlobalPlanner: # {{{
	def set_map(self, new_map):
		self.map = new_map
		self.grid = Grid(matrix=self.map)
		self.finder = AStarFinder(diagonal_movement=DiagonalMovement.never)

	def find_path(self, start, end):
		if (start == end).all():
			return []
		start = self.grid.node(start[1], start[0])
		end = self.grid.node(end[1], end[0])
		path, runs = self.finder.find_path(start, end, self.grid)
		self.grid.cleanup()
		if len(path) == 0:
			raise RuntimeError('Path not found')
		path = np.flip(path, axis=1)
		return path[1:]
		# don't need the first element, because its te current position of the agent
# }}}

class DiscreteEnv(gym.Env):
	def __init__(self, # {{{
		N,                # number of agents
		map_image,        # the image file to be used as a map
		):
		super(DiscreteEnv, self).__init__()

		self.N            = N
		self.map_image    = map_image

		## loading map
		self.map = cv2.imread(map_image, cv2.IMREAD_GRAYSCALE) / 255.
		if type(self.map) == type(None):
			raise Exception('Map image not found.')
		self.map = cv2.threshold(self.map, .5, 1, cv2.THRESH_BINARY)[1]
		self.map = self.map.astype(np.float32)

		assert(len(self.map.shape) == 2 and self.map.shape[0] == self.map.shape[1])
		self.map_size = self.map.shape[0]
		self.map_padded = pad_image(self.map)

		## constants
		self.RENDER_SIZE          = 700 # size of rendered image (square)
		self.curr_agent           = 0
		self.sf                   = self.RENDER_SIZE / self.map_size # scale factor, used at rendering
		self.ACTION_DELTAS        = np.array([[0, 1], [-1, 0], [0, -1], [1, 0], [0, 0]]) # movement for each act

		self.planner              = GlobalPlanner()
		self.planner.set_map(self.map)

		## generating colors for agents
		self.agent_colors = 1 - np.random.random(size=3*N).reshape((-1, 3))/2

		## observation space
		local_map     = spaces.Box(low=0, high=1, shape=(3, 3), dtype=int)
		agent_coords  = spaces.Box(low=0, high=self.map_size-2, shape=(self.N, 2), dtype=int)
		cost_mat      = spaces.Box(low=0, high=self.map_size*2, shape=(self.N, 3, 3), dtype=int)
		self.observation_space = spaces.Tuple((local_map, agent_coords, cost_mat))

		## action space
		self.action_space = spaces.Discrete(5)

		self.calculate_costmap()

		## generating starting states and goals
		# create a copy of the map, and mark where a new state or goal can be spawned
		# vector of indices where there are no walls, states or goals
		self.free_vec   = np.argwhere(self.map.astype(bool).flatten()).flatten()
		assert(self.free_vec.size >= self.N*2) # check if there is enough space for the agents and goals

		# generate states and goals
		self.states = np.empty((self.N, 2), dtype=int)
		self.goals  = np.empty((self.N, 2), dtype=int)
		for i in range(self.N):
			# states
			s = np.random.choice(self.free_vec)
			row = int(s // self.map_size)
			col =  s - row * self.map_size

			self.states[i] = np.array([row, col])
			self.free_vec = self.free_vec[self.free_vec != s]

			# goals
			g = np.random.choice(self.free_vec)
			row = int(g // self.map_size)
			col =  g - row * self.map_size

			self.goals[i] = np.array([row, col])
			self.free_vec = self.free_vec[self.free_vec != g]
	# }}}

	def calculate_costmap(self): # {{{
		self.cost_map             = np.empty((self.map_size,)*4, dtype=int) # this is global
		self.cost_map_local       = np.empty((self.N,3,3), dtype=int) # this is local
		for i1 in range(self.map_size):
			for i2 in range(self.map_size):
				for i3 in range(self.map_size):
					for i4 in range(self.map_size):
						if self.map[i1, i2] == 0 or self.map[i3, i4] == 0:
							self.cost_map[i1, i2, i3, i4] = -1
						else:
							pt_a = np.array([i1, i2])
							pt_b = np.array([i3, i4])
							path = self.planner.find_path(pt_a, pt_b)
							l    = len(path)
							self.cost_map[i1, i2, i3, i4] = l

		self.cost_map_padded = np.full((self.map_size+2,)*4, -1)
		self.cost_map_padded[1:-1, 1:-1, 1:-1, 1:-1] = self.cost_map.copy()
	# }}}

	def get_observation(self): # {{{
		p_v1 = self.states[self.curr_agent, 0] # horizontal, from
		p_v2 = p_v1 + 3                        # horizontal, to
		p_h1 = self.states[self.curr_agent, 1] # vertical, from
		p_h2 = p_h1 + 3                        # vertical, to
		local_map = self.map_padded[p_v1:p_v2, p_h1:p_h2].copy()

		agent_coords = np.roll(self.states.copy(), -self.curr_agent, axis=0)
		for n in range(1, self.N):
			agent_coords[n] -= agent_coords[0]

		# get costmaps
		for n in range(self.N):
			s = self.states[n]
			g = self.goals[n]
			self.cost_map_local[n] = self.cost_map_padded[s[0]:s[0]+3, s[1]:s[1]+3, g[0]+1, g[1]+1]
		self.cost_map_local = np.roll(self.cost_map_local.copy(), -self.curr_agent, axis=0)

		return local_map, agent_coords, self.cost_map_local
	# }}}

	def reset(self): # {{{
		return self.get_observation()
	# }}}

	def step(self, action): # {{{
		R = 0
		next_state = self.states[self.curr_agent] + self.ACTION_DELTAS[action]

		done = False
		hit = False
		if action != 4:
			# check for collision
			if -1 in next_state or self.map_size in next_state:
				# out of map
				hit = True
			elif self.map[tuple(next_state)] == 0:
				# hit wall
				hit = True
			elif len(np.flatnonzero((self.states == next_state).all(1))):
				# hit other agent
				hit = True

		if not hit and action != 4:
			# there is actual movement
			self.states[self.curr_agent] = next_state

			# check if goal is reached
			if (self.states[self.curr_agent] == self.goals[self.curr_agent]).all():
				done = True
				# generate new goal
				self.free_vec   = np.argwhere(self.map.astype(bool).flatten()).flatten()
				for n in range(self.N): # mark states and goals as occupied
					s = self.states[n]
					s_flat = int(s[0]*self.map_size + s[1])
					if s_flat in self.free_vec:
						self.free_vec = self.free_vec[self.free_vec != s_flat]

					g = self.goals[n]
					g_flat = int(g[0]*self.map_size + g[1])
					if g_flat in self.free_vec:
						self.free_vec = self.free_vec[self.free_vec != g_flat]

				g_new = np.random.choice(self.free_vec)
				row = int(g_new // self.map_size)
				col = g_new - row * self.map_size
				self.goals[self.curr_agent] = np.array([row, col])
		if not done:
			# no goal reaching reward
			R = -1

		self.curr_agent = (self.curr_agent+1) % self.N
		S = self.get_observation()

		return S, float(R), done, None
	# }}}

	def serialize(self, S, dtype=np.float32): # {{{
		# observations are tuples by default
		# this method created a 1D numpy array from that
		return np.hstack([s.reshape(-1) for s in S]).reshape(-1).astype(dtype)
	# }}}

	def deserialize(self, S, dtype=np.float32): # {{{
		local_map = S[:9].reshape((3,3))
		S = S[9:]
		agent_coords = S[:self.N*2].reshape((self.N, 2))
		S = S[self.N*2:]
		cost_map_local = S.reshape((-1, 3, 3))
		return local_map, agent_coords, cost_map_local
	# }}}

	def get_os_len(self): # {{{
		return (
			9 + # local_map
			self.N*2 + # agent_coords
			self.N*9 # cost_maps
			)
	# }}}

	def get_rendered_pic(self): # {{{
		# copy pic from map
		self.pic = self.map.copy().reshape(self.map_size, self.map_size, 1)
		self.pic = cv2.transpose(self.pic)
		self.pic = cv2.cvtColor(self.pic, cv2.COLOR_GRAY2RGB) # B&W to RGB
		self.pic = cv2.resize(self.pic, (self.RENDER_SIZE, self.RENDER_SIZE), 
			interpolation=cv2.INTER_AREA) # resize
		sf = self.sf # scale factor

		# draw goals
		for i, g in enumerate(self.goals):
			center = (int(g[0]*sf+sf/2), int(g[1]*sf+sf/2))
			radius = int(sf/2)
			p1 = (int(g[0]*sf), int(g[1]*sf))
			p2 = (int(g[0]*sf + sf), int(g[1]*sf + sf))
			c = tuple(self.agent_colors[i])
			self.pic = cv2.rectangle(self.pic, p1, p2, color=c, thickness=-1)

		# draw states
		for i, s in enumerate(self.states):
			center = (int(s[0]*sf+sf/2), int(s[1]*sf+sf/2))
			radius = int(sf/2 * .8)
			self.pic = cv2.circle(self.pic, center, radius, color=tuple(self.agent_colors[i]), thickness=-1)
			self.pic = cv2.circle(self.pic, center, radius, color=0, thickness=2)

		self.pic = cv2.transpose(self.pic)
		# different coordinate systems used by numpy and cv2

		for i, s in enumerate(self.states):
			# draw number
			font = cv2.FONT_HERSHEY_TRIPLEX
			scale = int(sf/40)
			thickness = int(sf/30)
			txt_size = cv2.getTextSize(str(i), font, scale, thickness)[0]

			org = (int((s[1]+.5)*sf - txt_size[1]//2), int((s[0]+.5)*sf + txt_size[0]//2))
			self.pic = cv2.putText(self.pic, str(i), org, font, scale, 0, thickness)

		return self.pic
	# }}}

	def render(self, timeout=0): # {{{
		cv2.imshow('MAPP environment', self.get_rendered_pic())
		cv2.waitKey(timeout)
	# }}}


if __name__ == '__main__': # {{{
	np.random.seed(0)

	N = 2

	env = DiscreteEnv(N, 'maps/test_4x4.jpg')
	S = env.reset()
	R = None

	while True:
		for n in range(N):
			A = env.action_space.sample()
			S_, R, done, _ = env.step(A)
			print('========================')
			print(f'N = {n}')
			print(f'S = {S}')
			print(f'A = {A}')
			print(f'R = {R}')
			print(f'done = {done}')
			env.render()
			S = S_
# }}}
