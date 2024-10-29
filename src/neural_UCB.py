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
		# Fetch data
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

import torch
import torch.nn as nn
import torch.optim as optim

class Network(nn.Module):
	def __init__(self, dim, hidden_size=100):
		super(Network, self).__init__()
		self.fc1 = nn.Linear(dim, hidden_size)
		self.activate = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, 1)
	def forward(self, x):
		return self.fc2(self.activate(self.fc1(x)))

class NeuralUCBDiag:
	def __init__(self, dim, lamdba=1, nu=1, hidden=100):
		self.func = Network(dim, hidden_size=hidden).cuda()
		self.context_list = []
		self.reward = []
		self.lamdba = lamdba
		self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
		self.U = lamdba * torch.ones((self.total_param,)).cuda()
		self.nu = nu

	def select(self, context):
		tensor = torch.from_numpy(context).float().cuda()
		mu = self.func(tensor)
		g_list = []
		sampled = []
		ave_sigma = 0
		ave_rew = 0
		for fx in mu:
			self.func.zero_grad()
			fx.backward(retain_graph=True)
			g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
			g_list.append(g)
			sigma2 = self.lamdba * self.nu * g * g / self.U
			sigma = torch.sqrt(torch.sum(sigma2))

			sample_r = fx.item() + sigma.item()

			sampled.append(sample_r)
			ave_sigma += sigma.item()
			ave_rew += sample_r
		arm = np.argmax(sampled)
		self.U += g_list[arm] * g_list[arm]
		return arm, g_list[arm].norm().item(), ave_sigma, ave_rew

	def train(self, context, reward):
		self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
		self.reward.append(reward)
		optimizer = optim.SGD(self.func.parameters(), lr=1e-2, weight_decay=self.lamdba)
		length = len(self.reward)
		index = np.arange(length)
		np.random.shuffle(index)
		cnt = 0
		tot_loss = 0
		while True:
			batch_loss = 0
			for idx in index:
				c = self.context_list[idx]
				r = self.reward[idx]
				optimizer.zero_grad()
				delta = self.func(c.cuda()) - r
				loss = delta * delta
				loss.backward()
				optimizer.step()
				batch_loss += loss.item()
				tot_loss += loss.item()
				cnt += 1
				if cnt >= 1000:
					return tot_loss / 1000
			if batch_loss / length <= 1e-3:
				return batch_loss / length

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--exp_idx", help="Index of experiment")
	parser.add_argument('--size', default=15000, type=int, help='bandit size')
	parser.add_argument('--dataset', default='cifar10', metavar='DATASET')
	parser.add_argument('--shuffle', type=bool, default=0, metavar='1 / 0', help='shuffle the data set or not')
	parser.add_argument('--seed', type=int, default=0, help='random seed for shuffle, 0 for None')
	parser.add_argument('--nu', type=float, default=1, metavar='v', help='nu for control variance')
	parser.add_argument('--lamdba', type=float, default=0.001, metavar='l', help='lambda for regularzation')
	parser.add_argument('--hidden', type=int, default=100, help='network hidden size')

	args = parser.parse_args()
	use_seed = None if args.seed == 0 else args.seed
	b = Bandit_multi(args.dataset, is_shuffle=args.shuffle, seed=use_seed)
	bandit_info = '{}'.format(args.dataset)
	l = NeuralUCBDiag(b.dim, args.lamdba, args.nu, args.hidden)
	ucb_info = '_{:.3e}_{:.3e}_{}'.format(args.lamdba, args.nu, args.hidden)

	regrets = []
	summ = 0
	for t in range(min(args.size, b.size)):
		context, rwd = b.step()
		arm_select, nrm, sig, ave_rwd = l.select(context)
		r = rwd[arm_select]
		reg = np.max(rwd) - r
		summ+=reg
		if t<10000:
			loss = l.train(context[arm_select], r)
		else:
			if t%10 == 0:
				loss = l.train(context[arm_select], r)
		regrets.append(summ)
		if t % 100 == 0:
			print('{}: {:.3f}, {:.3e}, {:.3e}, {:.3e}, {:.3e}'.format(t, summ, loss, nrm, sig, ave_rwd))
	
	path = "out/logs/cifar10/neural_UCB" + str(args.exp_idx)
	fr = open(path,'w')
	for i in regrets:
		fr.write(str(i))
		fr.write("\n")
	fr.close()
