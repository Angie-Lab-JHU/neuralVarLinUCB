from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import numpy as np
import argparse
import pickle
import os
import time

class Bandit_multi:
	def __init__(self, name, is_shuffle=True, seed=None):
		# X, y = fetch_openml('CIFAR_10', version=1, return_X_y=True)
		# with open('cifar10_x', 'wb') as fp:
		# 	pickle.dump(X, fp)
		# with open('cifar10_y', 'wb') as fp:
		# 	pickle.dump(y, fp)
		# quit()

		if name == 'cifar10':
			with open ('cifar10_x', 'rb') as fp:
				X = pickle.load(fp)
			with open ('cifar10_y', 'rb') as fp:
				y = pickle.load(fp)	
			X[np.isnan(X)] = - 1
			X = normalize(X)
		elif name == 'mnist':
			X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
			# avoid nan, set nan as -1
			X[np.isnan(X)] = - 1
			X = normalize(X)
		elif name == 'covertype':
			X, y = fetch_openml('covertype', version=3, return_X_y=True)
			# avoid nan, set nan as -1
			# X[np.isnan(X)] = - 1
			X = X.fillna(0)
			X = normalize(X)
		elif name == 'MagicTelescope':
			X, y = fetch_openml('MagicTelescope', version=1, return_X_y=True)
			# avoid nan, set nan as -1
			X[np.isnan(X)] = - 1
			X = normalize(X)
		elif name == 'shuttle':
			X, y = fetch_openml('shuttle', version=1, return_X_y=True)
			# avoid nan, set nan as -1
			X[np.isnan(X)] = - 1
			X = normalize(X)
		else:
			raise RuntimeError('Dataset does not exist')
		# Shuffle data
		if is_shuffle:
			self.X, self.y = shuffle(X, y, random_state=seed)
		else:
			self.X, self.y = X, y
		# generate one_hot coding:
		self.y_arm = OrdinalEncoder(
			dtype=int).fit_transform(self.y.values.reshape((-1, 1)))
		# cursor and other variables
		self.cursor = 0
		self.size = self.y.shape[0]
		self.n_arm = np.max(self.y_arm) + 1
		self.dim = self.X.shape[1] * self.n_arm
		self.act_dim = self.X.shape[1]

	def step(self):
		assert self.cursor < self.size
		X = np.zeros((self.n_arm, self.dim))
		for a in range(self.n_arm):
			X[a, a * self.act_dim:a * self.act_dim +
				self.act_dim] = self.X[self.cursor]
		arm = self.y_arm[self.cursor][0]
		rwd = np.zeros((self.n_arm,))
		rwd[arm] = 1
		self.cursor += 1
		return X, rwd

	def finish(self):
		return self.cursor == self.size

	def reset(self):
		self.cursor = 0

def inv_sherman_morrison(u, A_inv):
	"""Inverse of a matrix with rank 1 update.
	"""
	Au = np.dot(A_inv, u)
	A_inv -= np.outer(Au, Au)/(1+np.dot(u.T, Au))
	return A_inv

class LinearUCB:
	def __init__(self, dim, lamdba=1, nu=1):
		self.n_arm = 10
		self.lamdba = lamdba
		self.theta = np.random.uniform(-1, 1, (self.n_arm, dim))
		self.b = np.zeros((self.n_arm, dim))
		self.A_inv = np.array([np.eye(dim) for _ in range(self.n_arm)])

	def select(self, context):
		ucb = np.array([np.sqrt(np.dot(context[a,:], np.dot(self.A_inv[a], context[a,:].T))) for a in range(self.n_arm)])
		mu = np.array([np.dot(context[a,:], self.theta[a]) for a in range(self.n_arm)])
		arm = np.argmax(mu + self.lamdba * ucb)
		return arm

	def train(self, context, arm_select, reward):
		self.theta = np.array([np.matmul(self.A_inv[a], self.b[a]) for a in range(self.n_arm)])
		self.b[arm_select] += context[arm_select] * reward[arm_select]
		self.A_inv[arm_select] = inv_sherman_morrison(context[arm_select,:],self.A_inv[arm_select])

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--exp_idx", help="Index of experiment")
	parser.add_argument('--size', default=15000, type=int, help='bandit size')
	parser.add_argument('--dataset', default='cifar10', metavar='DATASET')
	parser.add_argument('--shuffle', type=bool, default=0, metavar='1 / 0', help='shuffle the data set or not')
	parser.add_argument('--seed', type=int, default=0, help='random seed for shuffle, 0 for None')
	parser.add_argument('--nu', type=float, default=1, metavar='v', help='nu for control variance')
	parser.add_argument('--lamdba', type=float, default=0.001, metavar='l', help='lambda for regularzation')

	args = parser.parse_args()
	use_seed = None if args.seed == 0 else args.seed
	b = Bandit_multi(args.dataset, is_shuffle=args.shuffle, seed=use_seed)
	bandit_info = '{}'.format(args.dataset)
	l = LinearUCB(b.dim, args.lamdba, args.nu)
	ucb_info = '_{:.3e}_{:.3e}'.format(args.lamdba, args.nu)

	regrets = []
	summ = 0
	for t in range(min(args.size, b.size)):
		context, rwd = b.step()
		arm_select = l.select(context)
		r = rwd[arm_select]
		reg = np.max(rwd) - r
		summ+=reg
		l.train(context, arm_select, rwd)
		regrets.append(summ)
		if t % 100 == 0:
			print('{}: {:.3f}'.format(t, summ))
	   
	path = "out/logs/cifar10/linear_UCB" + str(args.exp_idx)
	fr = open(path,'w')
	for i in regrets:
		fr.write(str(i))
		fr.write("\n")
	fr.close()
