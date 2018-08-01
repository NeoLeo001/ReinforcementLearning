from A3C import A3C
import gym 
import threading
import mxnet as mx
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

def thread_process(name):
	print name + ' starts ...'
	env = gym.make('CartPole-v0')
	global Tmax, T, lock, target_agent
	thread_agent = A3C(layers, hidden, actionspace, statespace)
	# trainer doesn't work here
	optimizer = mx.optimizer.Adam(learning_rate=0.001)

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
			a = target_agent.get_params()
			b = thread_agent.get_grads()
			for i in range(len(a)):
				optimizer.update(T, a[i], b[i], optimizer.create_state(T, a[i]))
			
			thread_agent.critic_backward()
			a = target_agent.get_params()
			b = thread_agent.get_grads()
			for i in range(len(a)):
				optimizer.update(T, a[i], b[i], optimizer.create_state(T, a[i]))
			
		except:
			print('Fail to get a lock!')
		finally:
			lock.release()

		thread_agent.copyto_params(target_agent.get_params())

	env.close()
	print(name + ' ends ...')


def target_run(name, show=False, train=False):
	print(name + ' starts ...')
	env = gym.make('CartPole-v0')
	global Tmax, T, lock, target_agent
	
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
	print(name + ' ends ...')

if __name__ == '__main__':
	main = threading.Thread(target=target_run, args=('Target',))
	main.start()
	for i in range(threads):
		t = threading.Thread(target=thread_process, args=('Subthreading %s'%str(i),))
		t.start()



