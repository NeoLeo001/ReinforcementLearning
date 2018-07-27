from policy import Approxmater
import numpy as np
import mxnet as mx
from mxnet import gluon

class MCPG(object):
	# optimizer, learning rate, activation
	# CarPole
	# adam, 0.001, tanh
	# adagrad, 0.01, tanh
	# adam, 0.001, relu
	# MountainCar

	def __init__(self, layers, hidden, actionspace, statespace, lr=0.01, dropout=0.1, activation='tanh', discount=1.0, *args, **kwargs):

		super(MCPG, self).__init__(*args, **kwargs)

		self.discount = discount
		self.actionspace = actionspace
		self.statespace = statespace
		self.policy = Approxmater(layers, hidden, actionspace, statespace, dropout, activation)
		self.policy.collect_params().initialize(mx.init.Xavier())
		self.episode_data = {
			'state':[],
			'action':[],
			'reward':[]
		}
		self.trainer = gluon.Trainer(self.policy.collect_params(), 'adagrad', {'learning_rate': lr, 'wd':0.01})

	def _reset_data(self):
		self.episode_data = {
			'state':[],
			'action':[],
			'reward':[]
		}

	def get_action(self, state):
		state = mx.nd.array(state).reshape((1,self.statespace))
		probs = np.squeeze(self.policy.forward(state).asnumpy())
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
				ret = self.episode_data['reward'][i] + self.discount*batch_data['return'][-1]
				batch_data['state'].append(self.episode_data['state'][i])
				batch_data['action'].append(self.episode_data['action'][i])
				batch_data['return'].append(ret)

			batch_data_s = mx.nd.array(batch_data['state'])
			batch_data_a = mx.nd.array(batch_data['action'])
			batch_data_r = mx.nd.array(batch_data['return'])

			with mx.autograd.record():
				probs = self.policy.forward(batch_data_s)
				action_prob = mx.nd.sum(probs*batch_data_a, axis=1).reshape((batch_size,))
				logprobs = (mx.nd.log(action_prob)*batch_data_r).reshape((batch_size,))
				loss = -mx.nd.sum(logprobs, axis=0).reshape((1,))
				loss.backward()
			self.trainer.step(batch_size)

			self._reset_data()




			
			

