from common.policy.basic_policy import BasicPolicy

def make_policy(env, policy_type):
	if policy_type == 'basic_policy':
		return BasicPolicy(env)

	raise ValueError(f'No such policy: {policy_type}')
