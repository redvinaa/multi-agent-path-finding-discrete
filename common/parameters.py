Params = {

	'1agent': {
		# environment
		'AGENTS': ['qlearning'],
		'MAP_IMAGE': 'maps/test_3x3.jpg',

		# learning
		'EPSILON_START': 1.,
		'EPSILON_FINAL': .0,
		'EPSILON_DECAY_LENGTH': 1e5,
		'GAMMA': .999,

		# ANN
		'HIDDEN_LAYER_SIZE': 20,
		'N_HIDDEN_LAYERS': 1,

		# other
		'STEPS': int(2e5),
		'LOAD_MODEL': '',
		'COMMENT': ''
	},

	'1agent-render': {
		# environment
		'AGENTS': ['qlearning'],
		'MAP_IMAGE': 'maps/test_3x3.jpg',

		# learning
		'EPSILON_START': .0,
		'EPSILON_FINAL': .0,
		'EPSILON_DECAY_LENGTH': 1e5,
		'GAMMA': .999,

		# ANN
		'HIDDEN_LAYER_SIZE': 20,
		'N_HIDDEN_LAYERS': 1,

		# other
		'STEPS': int(2e5),
		'LOAD_MODEL': '1agent',
		'COMMENT': ''
	},

	'2agent': {
		# environment
		'AGENTS': ['qlearning', 'qlearning'],
		'MAP_IMAGE': 'maps/test_3x3.jpg',

		# learning
		'EPSILON_START': 1.,
		'EPSILON_FINAL': .0,
		'EPSILON_DECAY_LENGTH': 4e5,
		'GAMMA': .9,

		# ANN
		'HIDDEN_LAYER_SIZE': 40,
		'N_HIDDEN_LAYERS': 1,

		# other
		'STEPS': int(6e5),
		'LOAD_MODEL': '',
		'COMMENT': ''
	},

}
