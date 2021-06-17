from common.env import DiscreteEnv
import argparse, yaml

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('names', default='default', nargs='+', help='Names of the parameter groups')
	args = parser.parse_args()

	with open('parameters.yaml', 'r') as f:
		Params = yaml.load(f, Loader=yaml.FullLoader)

	for name in args.names:
		params = Params[name]
		env = DiscreteEnv(sum(params['agents'].values()), params['map_image'])
		os_len = env.get_os_len()
		print(f'{name} > os_len = {os_len}')
