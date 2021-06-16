Params = {

	'1agent': {
		# environment
		'AGENTS': ['qlearning'],
		'MAP_IMAGE': 'maps/test_3x3.jpg',

		# learning
		'EPSILON_START': 1.,
		'EPSILON_FINAL': .0,
		'EPSILON_DECAY_LENGTH': 2e5,
		'GAMMA': .9,

		# ANN
		'HIDDEN_LAYER_SIZE': 30,
		'N_HIDDEN_LAYERS': 1,

		# other
		'STEPS': int(3e5),
		'LOAD_MODEL': '',
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
