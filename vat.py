# -*- coding: utf-8 -*-
import math
import numpy as np
import chainer, os, collections, six, math, random, copy
from chainer import cuda, Variable, serializers
from chainer import functions as F
import params
import sequential

class Object(object):
	pass

def to_object(dict):
	obj = Object()
	for key, value in dict.iteritems():
		setattr(obj, key, value)
	return obj

class Config(params.Params):
	def __init__(self):
		self.ndim_x = 28 * 28
		self.ndim_y = 10
		self.weight_init_std = 0.01
		self.weight_initializer = "Normal"		# Normal or GlorotNormal or HeNormal
		self.nonlinearity = "relu"
		self.optimizer = "Adam"
		self.learning_rate = 0.0001
		self.momentum = 0.9
		self.gradient_clipping = 10
		self.weight_decay = 0
		self.lambda_ = 1
		self.Ip = 1

class VAT(object):

	def __init__(self, params):
		super(VAT, self).__init__()
		self.params = copy.deepcopy(params)
		self.config = to_object(params["config"])
		config = self.config
		self.chain = sequential.chain.Chain()
		self.model = sequential.from_dict(params["model"])
		self.chain.add_sequence(self.model)
		self.chain.setup_optimizers(config.optimizer, config.learning_rate, config.momentum, config.weight_decay, config.gradient_clipping)
		self._gpu = False

	def load(self, dir=None):
		if dir is None:
			raise Exception()
		self.chain.load(dir + "/adgm.hdf5")

	def save(self, dir=None):
		if dir is None:
			raise Exception()
		try:
			os.mkdir(dir)
		except:
			pass
		self.chain.save(dir + "/adgm.hdf5")

	def backprop(self, loss):
		self.chain.backprop(loss)

	def to_gpu(self):
		self.chain.to_gpu()
		self._gpu = True

	@property
	def gpu_enabled(self):
		if cuda.available is False:
			return False
		return self._gpu

	@property
	def xp(self):
		if self.gpu_enabled:
			return cuda.cupy
		return np

	def to_variable(self, x):
		if isinstance(x, Variable) == False:
			x = Variable(x)
			if self.gpu_enabled:
				x.to_gpu()
		return x

	def to_numpy(self, x):
		if isinstance(x, Variable) == True:
			x.to_cpu()
			x = x.data
		if isinstance(x, cuda.ndarray) == True:
			x = cuda.to_cpu(x)
		return x

	def get_batchsize(self, x):
		return x.shape[0]

	def encode_x_y(self, x, apply_softmax=True, test=False):
		x = self.to_variable(x)
		y = self.model(x, test=test)
		if apply_softmax:
			return F.softmax(y)
		return y

	def argmax_x_label(self, x, test=False):
		y_distribution = self.to_numpy(self.encode_x_y(x, apply_softmax=True, test=test))
		return np.argmax(y_distribution, axis=1)

	def compute_kld(self, p, q):
		return F.reshape(F.sum(p * (F.log(p + 1e-16) - F.log(q + 1e-16)), axis=1), (-1, 1))

	def get_unit_vector(self, v):
		v /= (np.sqrt(np.sum(v ** 2, axis=1)).reshape((-1, 1)) + 1e-16)
		return v

	def compute_lds(self, x, xi=10, eps=1):
		x = self.to_variable(x)
		y1 = self.to_variable(self.encode_x_y(x, apply_softmax=True).data)		# unchain
		d = self.to_variable(self.get_unit_vector(np.random.normal(size=x.shape).astype(np.float32)))

		for i in xrange(self.config.Ip):
			y2 = self.encode_x_y(x + xi * d, apply_softmax=True)
			kld = F.sum(self.compute_kld(y1, y2))
			kld.backward()
			d = self.to_variable(self.get_unit_vector(self.to_numpy(d.grad)))
		
		y2 = self.encode_x_y(x + eps * d, apply_softmax=True)
		return -self.compute_kld(y1, y2)

