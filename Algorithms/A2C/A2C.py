from policy import Actor, Critic
import numpy as np
import mxnet as mx
from mxnet import gluon
import math

class A2C(object):
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
		self.actor_trainer = gluon.Trainer(self.actor.collect_params(), 'adam', {'learning_rate': 0.0001, 'wd':1e-4})
		self.critic_trainer = gluon.Trainer(self.critic.collect_params(), 'adam', {'learning_rate': 0.0001, 'wd':1e-4})

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
		probs, tmp = self.actor.forward_actor(state)
		probs = np.squeeze(probs.asnumpy())
		tmp = np.squeeze(tmp.asnumpy())
		# print '---------------params------------------'
		# print self.get_params()
		# print '---------------grads------------------'
		# print self.get_grads()
		if math.isnan(probs[0]):
			print('Detected NaN')
			print probs
			print tmp
			# print '---------------params------------------'
			# print self.get_params()
			# print '---------------grads------------------'
			# print self.get_grads()
			exit(0)
		# print probs
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
				# one-step foward: don't work in cartpole
				# sval = np.squeeze(self.critic.forward_critic(mx.nd.array(batch_data['state'][-1]).reshape((-1, self.statespace))).asnumpy())
				# 200-step forward: equal to MC
				sval = batch_data['return'][-1]
				ret = self.episode_data['reward'][i] + self.discount*sval
				batch_data['state'].append(self.episode_data['state'][i])
				batch_data['action'].append(self.episode_data['action'][i])
				batch_data['return'].append(ret)

			batch_data_s = mx.nd.array(batch_data['state'])
			batch_data_a = mx.nd.array(batch_data['action'])
			batch_data_r = mx.nd.array(batch_data['return'])
			batch_data_v = mx.nd.array(np.squeeze(self.critic.forward_critic(batch_data_s).asnumpy()))

			# print '-----------------------------------return-----------------------------------'
			# print batch_data['return']
			# print '-----------------------------------value-----------------------------------'
			# print np.squeeze(self.policy.forward_critic(batch_data_s).asnumpy())
			# print '-----------------------------------advantage-----------------------------------'
			# print batch_data['return']-np.squeeze(self.policy.forward_critic(batch_data_s).asnumpy())

			self.actor.collect_params().zero_grad()
			with mx.autograd.record():
				probs, _ = self.actor.forward_actor(batch_data_s)
				advs = batch_data_r - batch_data_v
				action_prob = mx.nd.sum(probs*batch_data_a, axis=1).reshape((batch_size,))
				logprobs = (mx.nd.log(action_prob)*advs).reshape((batch_size,))
				actor_loss = -mx.nd.sum(logprobs, axis=0).reshape((1,))

			actor_loss.backward()
			self.actor_trainer.step(batch_size, ignore_stale_grad=True)

			# print '---------------actor-params------------------'
			# print self.get_params()
			# print '---------------actor-grads------------------'
			# print self.get_grads()

			self.critic.collect_params().zero_grad()
			with mx.autograd.record():
				svals = self.critic.forward_critic(batch_data_s)
				advs = svals - batch_data_r
				critic_loss = mx.nd.sum(advs**2, axis=0).reshape((1,))
			
			critic_loss.backward()
			self.critic_trainer.step(batch_size, ignore_stale_grad=True)

			# print '---------------critic-params------------------'
			# print self.get_params()
			# print '---------------critic-grads------------------'
			# print self.get_grads()

			self._reset_data()

	def get_params(self):
		params_list = {
			'actor':[],
			'critic':[]
		}
		for name, value in self.actor.collect_params().items():
			params_list['actor'].append(value.data())

		for name, value in self.critic.collect_params().items():
			params_list['critic'].append(value.data())

		return params_list

	def copyto_params(self, params_list):
		actor_params = []
		for name, value in self.actor.collect_params().items():
			actor_params.append(value)

		assert len(params_list['actor']) == len(actor_params)
		for i in range(len(params_list['actor'])):
			actor_params[i].set_data(params_list['actor'][i])

		critic_params = []
		for name, value in self.critic.collect_params().items():
			critic_params.append(value)

		assert len(params_list['critic']) == len(critic_params)
		for i in range(len(params_list['critic'])):
			critic_params[i].set_data(params_list['critic'][i])

	def get_grads(self):
		grads_list = {
			'actor':[],
			'critic':[]
		}
		for name, value in self.actor.collect_params().items():
			if name.find('batchnorm') < 0:
				grads_list['actor'].append(value.grad())

		for name, value in self.critic.collect_params().items():
			if name.find('batchnorm') < 0:
				grads_list['critic'].append(value.grad())

		return grads_list

	def copyto_grads(self, grads_list):
		actor_values = []
		for name, value in self.actor.collect_params().items():
			if name.find('batchnorm') < 0:
				actor_values.append(value)

		assert len(grads_list['actor']) == len(actor_values)
		for i in range(len(grads_list['actor'])):
			actor_values[i]._grad = grads_list['actor'][i]

		critic_values = []
		for name, value in self.critic.collect_params().items():
			if name.find('batchnorm') < 0:
				critic_values.append(value)

		assert len(grads_list['critic']) == len(critic_values)
		for i in range(len(grads_list['critic'])):
			critic_values[i]._grad = grads_list['critic'][i]






			
			

