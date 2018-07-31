from policy import Approximator
import numpy as np
import mxnet as mx
from mxnet import gluon
import math

class A3C(object):
	# optimizer, actor learning rate, critic learning rate, activation
	# CarPole
	# adam, 0.001, 0.0001, tanh
	# MountainCar

	def __init__(self, 
				 layers, 
				 hidden, 
				 actionspace, 
				 statespace, 
				 lr=0.001, 
				 dropout=0.1, 
				 activation='tanh', 
				 discount=1.0, *args, **kwargs):

		super(A3C, self).__init__(*args, **kwargs)

		self.discount = discount
		self.actionspace = actionspace
		self.statespace = statespace
		self.turns = 0

		self.policy = Approximator(layers, hidden, actionspace, statespace, dropout, activation)
		self.policy.collect_params().initialize(mx.init.Xavier())

		self.episode_data = {
			'state':[],
			'action':[],
			'reward':[]
		}
		# self.trainer = gluon.Trainer(self.policy.collect_params(), 'adam', {'learning_rate': 0.0001, 'wd':1e-4, 'clip_gradient':1.0})
		# self.optimizer = mx.optimizer.Adam(learning_rate=0.001)

	def _reset_data(self):
		self.episode_data = {
			'state':[],
			'action':[],
			'reward':[]
		}
		self.turns = 0

	# the veriable of probs is very normal to nan. 
	# Find the reason!!!!!
	def get_action(self, state):
		state = mx.nd.array(state).reshape((1,self.statespace))
		probs = np.squeeze(self.policy.forward_actor(state).asnumpy())
		# print probs
		if math.isnan(probs[0]):
			print('Detected NaN')
			exit(0)
		index = np.random.choice(self.actionspace, p=probs)
		action = np.zeros((self.actionspace,))
		action[index] = 1
		return action, index

	def _feed(self, state, action, reward):
		self.episode_data['state'].append(state)
		self.episode_data['action'].append(action)
		self.episode_data['reward'].append(reward)

	def feed(self, state, action, reward, done):
		self._feed(state, action, reward)

		if done is True:
			time_steps = len(self.episode_data['state'])
			self.batch_size = time_steps

			batch_data = {
				'state':[],
				'action':[],
				'return':[]
			}
			batch_data['state'].append(self.episode_data['state'][-1])
			batch_data['action'].append(self.episode_data['action'][-1])
			batch_data['return'].append(self.episode_data['reward'][-1])

			for i in reversed(range(time_steps-1)):
				# one-step update
				# sval = np.squeeze(self.policy.forward_critic(mx.nd.array(batch_data['state'][-1]).reshape((-1, self.statespace))).asnumpy())
				# Monte Carlo update
				sval = batch_data['return'][-1]
				ret = self.episode_data['reward'][i] + self.discount*sval
				batch_data['state'].append(self.episode_data['state'][i])
				batch_data['action'].append(self.episode_data['action'][i])
				batch_data['return'].append(ret)

			self.batch_data_s = mx.nd.array(batch_data['state'])
			self.batch_data_a = mx.nd.array(batch_data['action'])
			self.batch_data_r = mx.nd.array(batch_data['return'])
			self.batch_data_v = mx.nd.array(np.squeeze(self.policy.forward_critic(self.batch_data_s).asnumpy()))

			self._reset_data()

	def actor_backward(self):
		self.policy.collect_params().zero_grad()
		with mx.autograd.record():
			probs = self.policy.forward_actor(self.batch_data_s)
			# svals = self.policy.forward_critic(batch_data_s)
			advs = self.batch_data_r - self.batch_data_v
			action_prob = mx.nd.sum(probs*self.batch_data_a, axis=1).reshape((self.batch_size,))
			logprobs = (mx.nd.log(action_prob)*advs).reshape((self.batch_size,))
			actor_loss = -1.0/self.batch_size*mx.nd.sum(logprobs, axis=0).reshape((1,))

		actor_loss.backward()

	def critic_backward(self):
		self.policy.collect_params().zero_grad()
		with mx.autograd.record():
			svals = self.policy.forward_critic(self.batch_data_s)
			advs = self.batch_data_r - svals
			critic_loss = 1.0/self.batch_size*mx.nd.sum(advs**2, axis=0).reshape((1,))

		critic_loss.backward()

	# index: count of update for the parameter
	# w: weight
	# g: grad
	# def update(self, index, w, g):
	# 	# self.trainer.step(1, ignore_stale_grad=True)
	# 	self.optimizer.update(index, w, g, self.optimizer.create_state(index, w))

	def get_params(self):
		params_list = []
		for name, value in self.policy.collect_params().items():
			params_list.append(value.data())
		return params_list

	def copyto_params(self, params_list):
		params = []
		for name, value in self.policy.collect_params().items():
			params.append(value)

		assert len(params_list) == len(params)
		for i in range(len(params_list)):
			params[i].set_data(mx.nd.array(params_list[i].asnumpy()))
			# print params[i].name

	def get_grads(self):
		grads_list = []
		for name, value in self.policy.collect_params().items():
			if name.find('batchnorm') < 0:
				grads_list.append(mx.nd.array(value.grad().asnumpy()))
		return grads_list

	def copyto_grads(self, grads_list):
		values = []
		for name, value in self.policy.collect_params().items():
			if name.find('batchnorm') < 0:
				values.append(value)

		assert len(grads_list) == len(values)
		for i in range(len(grads_list)):
			for arr in values[i]._check_and_get(values[i]._grad, list):
				arr[:] = grads_list[i]





			
			

