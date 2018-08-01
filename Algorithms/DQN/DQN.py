from policy import Approxmater
import numpy as np
import mxnet as mx
from mxnet import gluon
from replay_memory import ReplayMemory

class DQN(object):
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
				 discount=0.8, 
				 epsilon=0.9, 
				 epsilon_wd=0.001,
				 memory=10000,
				 start_turn=100,
				 batch_size=32,
				 update_period=100,
				  *args, **kwargs):

		super(DQN, self).__init__(*args, **kwargs)

		self.discount = discount
		self.actionspace = actionspace
		self.statespace = statespace
		self.epsilon = epsilon
		self.epsilon_wd = epsilon_wd
		self.start_turn = start_turn # start to train when size of the replay memory reaches to 'start_turn'
		self.batch_size = batch_size
		self.update_period = update_period
		assert start_turn > batch_size
		self.policy = Approxmater(layers, hidden, actionspace, statespace, dropout, activation)
		self.target_policy = Approxmater(layers, hidden, actionspace, statespace, dropout, activation)
		self.policy.collect_params().initialize(mx.init.Xavier())
		self.target_policy.collect_params().initialize(mx.init.Xavier())

		self.trainer = gluon.Trainer(self.policy.collect_params(), 'adagrad', {'learning_rate': lr})
		self.replayMemory = ReplayMemory(memory, actionspace, statespace)
		self.turn = 0
		self._copyto_target()


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

	def _feed(self, state, action, reward, nextstate):
		self.replayMemory.add(state, action, reward, nextstate)

	def _copyto_target(self):
		params = []
		target_params = []
		for name, value in self.policy.collect_params().items():
			params.append(mx.nd.array(np.squeeze(value.data().asnumpy())))
		for name, value in self.target_policy.collect_params().items():
			target_params.append(value)

		assert len(params) == len(target_params)

		for i in range(len(params)):
			target_params[i].set_data(params[i])

	def train(self, state, action, reward, nextstate):
		self._feed(state, action, reward, nextstate)
		self.turn += 1

		if self.replayMemory.size() > self.start_turn:

			batch_data = {
				'state':[],
				'action':[],
				'return':[]
			}

			memory_batch_data = self.replayMemory.get_minibatch(self.batch_size)
			next_maxqs = []
			for i in range(len(memory_batch_data['batch_nextstates'])):
				if memory_batch_data['batch_nextstates'][i] is None:
					next_maxqs.append(.0)
				else:
					next_qvals = self.target_policy.forward(mx.nd.array(memory_batch_data['batch_nextstates'][i]).reshape((1,self.statespace)))
					next_maxqs.append(np.max(np.squeeze(next_qvals.asnumpy())))

			rets = np.array(memory_batch_data['batch_rewards']) + self.discount*np.array(next_maxqs)
			batch_data['state'] = memory_batch_data['batch_states']
			batch_data['action'] = memory_batch_data['batch_actions']
			batch_data['return'] = rets

			# mx.nd.squeeze hasn't been supported.
			batch_data_s = mx.nd.array(batch_data['state'])
			batch_data_a = mx.nd.array(batch_data['action'])
			batch_data_r = mx.nd.array(batch_data['return'])

			with mx.autograd.record():
				qvals = self.policy.forward(batch_data_s)
				action_qvals = mx.nd.sum(qvals*batch_data_a, axis=1).reshape((self.batch_size,))
				sqrerror = ((action_qvals-batch_data_r)**2).reshape((self.batch_size,))
				loss = -mx.nd.sum(sqrerror, axis=0).reshape((1,))
				loss.backward()
			self.trainer.step(self.batch_size)

		if self.turn % self.update_period:
			self._copyto_target()