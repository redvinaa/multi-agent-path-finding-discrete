from common import DiscreteEnv, Params
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('names', default='default', nargs='+', help='Names of the parameter groups')
	args = parser.parse_args()

	if len(args.names) == 1:
		params = Params[args.names[0]]
		env = DiscreteEnv(len(params['AGENTS']), params['MAP_IMAGE'])
		os_len = env.get_os_len()
		print(f'os_len = {os_len}')

	else:
		for name in args.names:
			params = Params[name]
			env = DiscreteEnv(len(params['AGENTS']), params['MAP_IMAGE'])
			os_len = env.get_os_len()
			print(f'{name} > os_len = {os_len}')
