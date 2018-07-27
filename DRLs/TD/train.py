from TD import TD
import gym 
env = gym.make('CartPole-v0')
# env = gym.make('MountainCar-v0')
layers = 4
hidden = 16
actionspace = 2#env.action_space 3
statespace = 4#env.observation_space 2


def run(max_episode, show=False, train=False):
	agent = TD(layers, hidden, actionspace, statespace)

	episode = 0
	while True:
		episode += 1
		if episode > max_episode:
			break

		state = env.reset()
		done = False
		count = 0
		while done is False:
			pre_state = state
			action, index = agent.get_action(state)
			state, r, done, _ = env.step(index)
			if train:
				agent.train(pre_state, action, r, done)

			count += 1
		print count

	env.close()

if __name__ == '__main__':
	run(10000, show=True, train=True)



