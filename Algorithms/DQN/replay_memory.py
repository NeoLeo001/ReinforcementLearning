import numpy as np
import random

class ReplayMemory(object):
	def __init__(self, memory, actionspace, statespace, **kwargs):
		super(ReplayMemory, self).__init__(**kwargs)
		self.states = [None]*memory
		self.actions = [None]*memory
		self.nextstates = [None]*memory
		self.rewards = [None]*memory
		self.top = -1
		self.memory = memory

	def add(self, state, action, reward, nextstate):
		self.top += 1
		addindx = self.top%self.memory
		self.actions[addindx] = action
		self.states[addindx] = state
		self.rewards[addindx] = reward
		self.nextstates[addindx] = nextstate

	def get_minibatch(self, batch_size):
		minibatch = {
			'batch_states':[],
			'batch_actions':[],
			'batch_rewards':[],
			'batch_nextstates':[]
		}

		assert self.top > batch_size
		sample_indx = random.sample(range(self.top), batch_size)

		for i in sample_indx:
			index = i%self.memory
			minibatch['batch_states'].append(self.states[index])
			minibatch['batch_actions'].append(self.actions[index])
			minibatch['batch_rewards'].append(self.rewards[index])
			minibatch['batch_nextstates'].append(self.nextstates[index])


		return minibatch

	def size(self):
		return self.top

	def clear(self):
		self.top = -1
