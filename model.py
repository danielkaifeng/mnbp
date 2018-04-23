
import tensorflow as tf

import numpy as np


def my_sigmoid_loss(labels, logits):
		#relu_logits = tf.nn.relu(logits)
		#neg_abs_logits = -tf.abs(logits)		
		#res = tf.add(relu_logits - logits * labels, tf.log1p(tf.exp(neg_abs_logits)))

		gamma = 1.15 *labels - 0.15
		#gamma = 2 *labels - 1
		res = 1 - tf.log1p( gamma*logits/(1+ tf.abs(logits)) )
		return res

def leaky_relu(x, leak=0.001, name='leaky_relu'):
		return tf.maximum(x, x * leak, name=name)

def batch_norm(x, phase_train, scope='bn'):
	n_out = x.get_shape().as_list()[-1]
	with tf.variable_scope(scope):
		beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
									 name='beta', trainable=True)
		gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
									  name='gamma', trainable=True)
		batch_mean, batch_var = tf.nn.moments(x, [0,1], name='moments')
		ema = tf.train.ExponentialMovingAverage(decay=0.5)

		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

		mean, var = tf.cond(phase_train, mean_var_with_update,
							lambda: (ema.average(batch_mean), ema.average(batch_var)))
		normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
	return normed



class build_network():
	def __init__(self):
		self.x = tf.placeholder(tf.float32, shape=[None, 159+299], name='x')
		self.labels = tf.placeholder(tf.float32, [None, 5], name='y')
		self.is_train = tf.placeholder(tf.bool, name='is_train')

		self.dropout = tf.placeholder(tf.float32, name='dropout')

		with tf.name_scope("cal_loss") as scope:
				logits, gamma = self.network(self.x, self.dropout, self.is_train) 
				
				#base = np.array([126,78,1.6,1.3,2.7])
				base = np.array([120, 75, 2, 2, 3])

				top = np.array([260,130,30, 6, 15])
				low = np.array([69,37, 0.05, 0.05, 0.05])
				self.mid = logits
				logits += base
	
				#logits = tf.maximum(logits, low)
				logits = tf.nn.relu(logits)
				self.logits = tf.minimum(logits, top)

				opt = tf.train.AdamOptimizer(0.001)

				alpha = np.array([100, 100, 1, 1, 1])
				#x = tf.clip_by_value(self.labels / alpha, 1e-4, 1 - 1e-4)
				#y = tf.clip_by_value(self.logits / alpha, 1e-4, 1 - 1e-4)
				x = self.logits / alpha
				y = self.labels / alpha

				#mse = tf.sqrt(tf.reduce_mean(tf.square(self.mid)))
				#self.loss = tf.reduce_mean(tf.log(tf.square(self.logits-self.labels) + 0.00001 )) 
				self.loss = tf.reduce_mean(tf.squared_difference(x, y))
				#self.acc2 = tf.reduce_mean(np.array([1,1,2,2,1]) * tf.square(tf.log1p(logits) - tf.log1p(self.labels)))
				#self.acc = tf.reduce_mean(tf.square(tf.log1p(logits) - tf.log1p(self.labels)))

				#tf.losses.mean_squared_error(labels, predictions)
				self.optimizer = opt.minimize(self.loss)
		
		
	def network(self, x, dropout, is_train):

		with tf.name_scope("mlp_network") as scope:
				x1 = tf.layers.dense(x[:,0:159], 128)
				x2 = tf.layers.dense(x[:,159:458], 128)
				x = tf.concat([x1,x2], 1)
				x = tf.expand_dims(x,2)

				for n in range(3):
					conv1 = tf.layers.conv1d(x, filters = 64, kernel_size=7, padding="same", activation=tf.nn.relu)
					conv2 = tf.layers.conv1d(x, filters = 64, kernel_size=5, padding="same", activation=tf.nn.relu)
					conv3 = tf.layers.conv1d(x, filters = 64, kernel_size=1, padding="same", activation=tf.nn.relu)
					x = tf.concat([conv1, conv2, conv3], 2)
					x = tf.layers.max_pooling1d(x, pool_size=2, strides=2, padding="same")
					x = batch_norm(x, is_train, scope='bn_%d' % n)

					x = tf.nn.relu(x)
					x = tf.layers.dropout(x, rate=self.dropout)
					


				x = tf.layers.dense(x, 64)
				x = tf.reshape(x, [-1, 32*64])
				x = tf.nn.relu(x)
				x = tf.layers.dropout(x, rate=self.dropout)
				x = tf.layers.dense(x, 32)
				x = tf.nn.relu(x)
				x = tf.layers.dropout(x, rate=self.dropout)

				logits = tf.layers.dense(x, units=5)
				gamma = tf.layers.dense(x, units=5)
				gamma = tf.log1p(tf.abs(gamma)) + 1
		return logits, gamma


