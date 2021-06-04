Params = {

	'1agent-st1': {
		# environment
		'N_AGENTS': 1,
		'MAP_IMAGE': 'maps/test_4x4.jpg',

		# learning
		'EPSILON_START': 1.,
		'EPSILON_FINAL': .0,
		'EPSILON_DECAY_LENGTH': 5e4,
		'GAMMA': .9,

		# ANN
		'HIDDEN_LAYER_SIZE': 5,
		'N_HIDDEN_LAYERS': 1,

		# other
		'STEPS': int(5e4),
		'STAGE': 1,
		'MODEL_SAVE_FREQ': 0,
		'LOAD_MODEL': '',
		'COMMENT': ''
	},

	'1agent-st2': {
		# environment
		'N_AGENTS': 1,
		'MAP_IMAGE': 'maps/test_4x4.jpg',

		# learning
		'EPSILON_START': 1.,
		'EPSILON_FINAL': .0,
		'EPSILON_DECAY_LENGTH': 5e4,
		'GAMMA': .9,

		# ANN
		'HIDDEN_LAYER_SIZE': 5,
		'N_HIDDEN_LAYERS': 1,

		# other
		'STEPS': int(5e4),
		'STAGE': 2,
		'MODEL_SAVE_FREQ': 0,
		'LOAD_MODEL': '',
		'COMMENT': ''
	},

	'2agent': {
		# environment
		'N_AGENTS': 2,
		'MAP_IMAGE': 'maps/test_4x4.jpg',

		# learning
		'EPSILON_START': 1.,
		'EPSILON_FINAL': .1,
		'EPSILON_DECAY_LENGTH': 1e5,
		'GAMMA': .9,

		# ANN
		'HIDDEN_LAYER_SIZE': 10,
		'N_HIDDEN_LAYERS': 1,

		# other
		'STEPS': int(2e5),
		'STAGE': 2,
		'MODEL_SAVE_FREQ': 0,
		'LOAD_MODEL': '',
		'COMMENT': ''
	},

	'2agent-1': {
		# environment
		'N_AGENTS': 2,
		'MAP_IMAGE': 'maps/test_4x4.jpg',

		# learning
		'EPSILON_START': 1.,
		'EPSILON_FINAL': .1,
		'EPSILON_DECAY_LENGTH': 1e5,
		'GAMMA': .9,

		# ANN
		'HIDDEN_LAYER_SIZE': 10,
		'N_HIDDEN_LAYERS': 1,

		# other
		'STEPS': int(2e5),
		'STAGE': 2,
		'MODEL_SAVE_FREQ': 0,
		'LOAD_MODEL': '',
		'COMMENT': ''
	},

	'2agent-2': {
		# environment
		'N_AGENTS': 2,
		'MAP_IMAGE': 'maps/test_4x4.jpg',

		# learning
		'EPSILON_START': 1.,
		'EPSILON_FINAL': .1,
		'EPSILON_DECAY_LENGTH': 1e5,
		'GAMMA': .9,
		'LR': 1e-3,
		'MOMENTUM': 0.1,

		# ANN
		'HIDDEN_LAYER_SIZE': 10,
		'N_HIDDEN_LAYERS': 1,

		# other
		'STEPS': int(1e5),
		'STAGE': 2,
		'MODEL_SAVE_FREQ': 0,
		'LOAD_MODEL': '',
		'COMMENT': ''
	},

}
