import mxnet as mx
from mxnet import gluon, nd

class Approximator(gluon.nn.Block):
	def __init__(self, layers, hidden, actionspace, statespace, dropout=0.1, activation='relu', *args, **kwargs):
		gluon.nn.Block.__init__(self)
		self.layers = layers
		self.hidden = hidden
		self.actionspace = actionspace
		self.statespace = statespace
		self.activation = activation
		self.dropout = dropout

		with self.name_scope():
			self.net1 = gluon.nn.Sequential()
			with self.net1.name_scope():
				self.net1.add(gluon.nn.Dense(units=self.hidden, in_units=self.statespace, activation=self.activation))
				# self.net1.add(gluon.nn.BatchNorm(axis=-1))
				self.net1.add(gluon.nn.Dropout(self.dropout))
				if self.layers > 2:
					for i in range(self.layers-2):
						self.net1.add(gluon.nn.Dense(units=self.hidden, in_units=self.hidden, activation=self.activation))
						# self.net1.add(gluon.nn.BatchNorm(axis=-1))
						self.net1.add(gluon.nn.Dropout(self.dropout))

			self.net2 = gluon.nn.Sequential()
			with self.net2.name_scope():
				self.net2.add(gluon.nn.Dense(units=self.hidden, in_units=self.statespace, activation=self.activation))
				# self.net2.add(gluon.nn.BatchNorm(axis=-1))
				self.net2.add(gluon.nn.Dropout(self.dropout))
				if self.layers > 2:
					for i in range(self.layers-2):
						self.net2.add(gluon.nn.Dense(units=self.hidden, in_units=self.hidden, activation=self.activation))
						# self.net2.add(gluon.nn.BatchNorm(axis=-1))
						self.net2.add(gluon.nn.Dropout(self.dropout))

			self.actor = gluon.nn.Dense(units=self.actionspace, in_units=self.hidden, activation=self.activation, prefix='actor')
			self.critic = gluon.nn.Dense(units=1, in_units=self.hidden, activation='sigmoid', prefix='critic')

	def forward_actor(self, input):
		'''
		shape of input: batchsize * statespace
		'''
		tmp = self.actor(self.net1(input))
		probs = mx.nd.softmax(tmp, axis=1)

		return probs

	def forward_critic(self, input):
		'''
		shape of input: batchsize * statespace
		'''
		sval = self.critic(self.net2(input))

		return sval



