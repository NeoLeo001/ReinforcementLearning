from A3C import A3C
import gym 
import threading
# env = gym.make('MountainCar-v0')
layers = 4
hidden = 16
actionspace = 2#env.action_space 3
statespace = 4#env.observation_space 2
Tmax = 10000
threads = 4
T = 0
lock = threading.Lock()

target_agent = A3C(layers, hidden, actionspace, statespace)

def thread_process():
	env = gym.make('CartPole-v0')
	global Tmax, T, lock, target_agent
	thread_agent = A3C(layers, hidden, actionspace, statespace)

	while Tmax > T:

		state = env.reset()
		done = False
		count = 0
		while done is False:
			pre_state = state
			action, index = thread_agent.get_action(state)
			state, r, done, _ = env.step(index)

			thread_agent.feed(pre_state, action, r/200.0, done)
		T += 1

		lock.acquire()
		try:
			thread_agent.actor_backward()
			target_agent.copyto_grads(thread_agent.get_grads())
			target_agent.update()
			thread_agent.critic_backward()
			target_agent.copyto_grads(thread_agent.get_grads())
			target_agent.update()
			target_agent.policy.collect_params().zero_grad()
		finally:
			lock.release()

		thread_agent.policy.collect_params().zero_grad()
		thread_agent.copyto_params(target_agent.get_params())

	env.close()


def target_run(show=False, train=False):
	env = gym.make('CartPole-v0')
	global Tmax, T, target_agent, lock
	
	while Tmax > T:

		state = env.reset()
		done = False
		count = 0
		while done is False:
			pre_state = state
			lock.acquire()
			try:
				action, index = target_agent.get_action(state)
			finally:
				lock.release()
			state, r, done, _ = env.step(index)

			count += 1
		print count

	env.close()

if __name__ == '__main__':
	main = threading.Thread(target=target_run)
	main.start()
	for i in range(threads):
		t = threading.Thread(target=thread_process)
		t.start()



