from policy import Actor, Critic
import numpy as np
import mxnet as mx
from mxnet import gluon

class A2C(object):
	# optimizer, actor learning rate, critic learning rate, activation
	# CarPole
	# adam, 0.001, 0.0001, tanh
	# MountainCar

	def __init__(self, layers, hidden, actionspace, statespace, lr=0.001, dropout=0.1, activation='tanh', discount=1.0, *args, **kwargs):

		super(A2C, self).__init__(*args, **kwargs)

		self.discount = discount
		self.actionspace = actionspace
		self.statespace = statespace

		self.actor = Actor(layers, hidden, actionspace, statespace, dropout, activation)
		self.actor.collect_params().initialize(mx.init.Xavier())
		self.critic = Critic(layers, hidden, actionspace, statespace, dropout, activation)
		self.critic.collect_params().initialize(mx.init.Xavier())

		self.episode_data = {
			'state':[],
			'action':[],
			'reward':[]
		}
		self.actor_trainer = gluon.Trainer(self.actor.collect_params(), 'adam', {'learning_rate': 0.00002, 'wd':0.01})
		self.critic_trainer = gluon.Trainer(self.critic.collect_params(), 'adam', {'learning_rate': 0.00005, 'wd':0.01})

	def _reset_data(self):
		self.episode_data = {
			'state':[],
			'action':[],
			'reward':[]
		}

	# the veriable of probs is very normal to num. 
	# Find the reason!!!!!
	def get_action(self, state):
		state = mx.nd.array(state).reshape((1,self.statespace))
		probs = np.squeeze(self.actor.forward(state).asnumpy())
		index = np.random.choice(self.actionspace, p=probs)
		action = np.zeros((self.actionspace,))
		action[index] = 1
		return action, index

	def _feed(self, state, action, reward):
		self.episode_data['state'].append(state)
		self.episode_data['action'].append(action)
		self.episode_data['reward'].append(reward)


	def train(self, state, action, reward, done):
		self._feed(state, action, reward)

		if done is True:
			time_steps = len(self.episode_data['state'])
			batch_size = time_steps

			batch_data = {
				'state':[],
				'action':[],
				'return':[]
			}
			batch_data['state'].append(self.episode_data['state'][-1])
			batch_data['action'].append(self.episode_data['action'][-1])
			batch_data['return'].append(self.episode_data['reward'][-1])

			for i in reversed(range(time_steps-1)):
				sval = np.squeeze(self.critic.forward(mx.nd.array(batch_data['state'][-1]).reshape((-1, self.statespace))).asnumpy())
				ret = self.episode_data['reward'][i] + self.discount*sval
				batch_data['state'].append(self.episode_data['state'][i])
				batch_data['action'].append(self.episode_data['action'][i])
				batch_data['return'].append(ret)

			batch_data_s = mx.nd.array(batch_data['state'])
			batch_data_a = mx.nd.array(batch_data['action'])
			batch_data_r = mx.nd.array(batch_data['return'])

			with mx.autograd.record():
				probs = self.actor.forward(batch_data_s)
				svals = self.critic.forward(batch_data_s)
				advs = batch_data_r - svals
				action_prob = mx.nd.sum(probs*batch_data_a, axis=1).reshape((batch_size,))
				logprobs = (mx.nd.log(action_prob)*advs).reshape((batch_size,))
				actor_loss = -mx.nd.sum(logprobs, axis=0).reshape((1,))
				actor_loss.backward()

			self.actor_trainer.step(batch_size)

			with mx.autograd.record():
				svals = self.critic.forward(batch_data_s)
				advs = batch_data_r - svals
				critic_loss = mx.nd.sum(advs**2, axis=0).reshape((1,))
				critic_loss.backward()

			self.critic_trainer.step(batch_size)

			self._reset_data()




			
			

