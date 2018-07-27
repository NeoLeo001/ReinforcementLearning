from policy import Approxmater
import numpy as np
import mxnet as mx
from mxnet import gluon

class TD(object):
	# optimizer, learning rate, activation, discount
	# CarPole
	# adam, 0.00001, tanh, 0.9
	# adagrad, 0.00001, tanh, 0.9
	# MountainCar
	# -,-,-

	def __init__(self, 
				 layers, 
				 hidden, 
				 actionspace, 
				 statespace, 
				 lr=0.00001, 
				 dropout=0.1, 
				 activation='tanh', 
				 discount=0.9, 
				 epsilon=0.9, 
				 epsilon_wd=0.0001, *args, **kwargs):

		super(TD, self).__init__(*args, **kwargs)

		self.discount = discount
		self.actionspace = actionspace
		self.statespace = statespace
		self.epsilon = epsilon
		self.epsilon_wd = epsilon_wd
		self.policy = Approxmater(layers, hidden, actionspace, statespace, dropout, activation)
		self.policy.collect_params().initialize(mx.init.Xavier())
		self.episode_data = {
			'state':[],
			'action':[],
			'reward':[]
		}
		self.trainer = gluon.Trainer(self.policy.collect_params(), 'adagrad', {'learning_rate': lr, 'wd':0.001})

	def _reset_data(self):
		self.episode_data = {
			'state':[],
			'action':[],
			'reward':[]
		}

	def get_action(self, state):
		# trade off between exploration and exploitation using epsilon-greedy approach
		if self.epsilon > 1e-3:
			rand = np.random.choice([True, False], p=[self.epsilon, 1-self.epsilon])
			if rand:
				index = np.random.choice(self.actionspace)
				action = np.zeros((self.actionspace,))
				action[index] = 1
			else:
				state = mx.nd.array(state).reshape((1,self.statespace))
				qvals = np.squeeze(self.policy.forward(state).asnumpy())
				index = np.argmax(qvals)
				action = np.zeros((self.actionspace,))
				action[index] = 1
			self.epsilon -= self.epsilon_wd
		else:
			state = mx.nd.array(state).reshape((1,self.statespace))
			qvals = np.squeeze(self.policy.forward(state).asnumpy())
			index = np.argmax(qvals)
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

				next_qvals = self.policy.forward(mx.nd.array(batch_data['state'][-1]).reshape((1,self.statespace)))
				next_maxq = np.max(np.squeeze(next_qvals.asnumpy()))
				ret = self.episode_data['reward'][i] + self.discount*next_maxq
				batch_data['state'].append(self.episode_data['state'][i])
				batch_data['action'].append(self.episode_data['action'][i])
				batch_data['return'].append(ret)

			batch_data_s = mx.nd.array(batch_data['state'])
			batch_data_a = mx.nd.array(batch_data['action'])
			batch_data_r = mx.nd.array(batch_data['return'])

			with mx.autograd.record():
				qvals = self.policy.forward(batch_data_s)
				action_qvals = mx.nd.sum(qvals*batch_data_a, axis=1).reshape((batch_size,))
				sqrerror = ((action_qvals-batch_data_r)**2).reshape((batch_size,))
				loss = -mx.nd.sum(sqrerror, axis=0).reshape((1,))
				loss.backward()
			self.trainer.step(batch_size)

			self._reset_data()




			
			

