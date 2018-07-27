import mxnet as mx
from mxnet import gluon, nd

class Approxmater(gluon.nn.Block):
	def __init__(self, layers, hidden, actionspace, statespace, dropout=0.1, activation='relu', *args, **kwargs):
		gluon.nn.Block.__init__(self)
		self.layers = layers
		self.hidden = hidden
		self.actionspace = actionspace
		self.statespace = statespace
		self.activation = activation
		self.dropout = dropout

		with self.name_scope():
			self.net = gluon.nn.Sequential()
			with self.net.name_scope():
				self.net.add(gluon.nn.Dense(units=self.hidden, in_units=self.statespace, activation=self.activation))
				self.net.add(gluon.nn.Dropout(self.dropout))
				if self.layers > 2:
					for i in range(self.layers-2):
						self.net.add(gluon.nn.Dense(units=self.hidden, in_units=self.hidden, activation=self.activation))
						self.net.add(gluon.nn.Dropout(self.dropout))

				self.net.add(gluon.nn.Dense(units=self.actionspace, in_units=self.hidden, activation=self.activation))

	def forward(self, input):
		'''
		shape of input: batchsize * statespace
		'''
		qvals = self.net(input)

		return qvals

