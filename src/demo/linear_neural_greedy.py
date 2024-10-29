from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import numpy as np
import argparse
import pickle
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

def inv_sherman_morrison(u, A_inv):
	"""Inverse of a matrix with rank 1 update.
	"""
	Au = np.dot(A_inv, u)
	A_inv -= np.outer(Au, Au)/(1+np.dot(u.T, Au))
	return A_inv

class Network(nn.Module):
	def __init__(self, dim, hidden_size=100):
		super(Network, self).__init__()
		self.fc1 = nn.Linear(dim, hidden_size)
		self.activate = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, 20)
	def forward(self, x):
		return self.fc2(self.activate(self.fc1(x)))

class NeuralLinearUCB:
	def __init__(self, dim, lamdba=1, nu=1, hidden=100):
		self.n_arm = 4
		self.func = Network(dim, hidden_size=hidden).cuda()
		self.context_list = []
		self.arm_list = []
		self.reward = []
		self.lamdba = lamdba
		self.theta = np.random.uniform(-1, 1, (self.n_arm, dim))
		self.b = np.zeros((self.n_arm, dim))
		self.A_inv = np.array([np.eye(dim) for _ in range(self.n_arm)])

	def select(self, context):
		tensor = torch.from_numpy(context).float().cuda()
		features = self.func(tensor).cpu().detach().numpy()
		mu = np.array([np.dot(features[a,:], self.theta[a]) for a in range(self.n_arm)])
		arm = np.argmax(mu)
		return arm

	def train(self, context, arm_select, reward):
		self.context_list.append(torch.from_numpy(context[arm_select].reshape(1, -1)).float())
		self.arm_list.append(arm_select)
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
				a = self.arm_list[idx]
				r = self.reward[idx]
				optimizer.zero_grad()
				features = self.func(c.cuda())
				mu = (features * torch.from_numpy(self.theta[a]).float().cuda()).sum(dim=1, keepdims=True)
				delta = mu - r
				loss = delta * delta
				loss.backward()
				optimizer.step()
				batch_loss += loss.item()
				tot_loss += loss.item()
				cnt += 1
				if cnt >= 1000:
					return tot_loss / 1000

	def update_model(self, context, arm_select, reward):
		tensor = torch.from_numpy(context).float().cuda()
		context = self.func(tensor).cpu().detach().numpy()
		self.theta = np.array([np.matmul(self.A_inv[a], self.b[a]) for a in range(self.n_arm)])
		self.b[arm_select] += context[arm_select] * reward[arm_select]
		self.A_inv[arm_select] = inv_sherman_morrison(context[arm_select,:],self.A_inv[arm_select])


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	with open ('contexts', 'rb') as fp:
		contexts = pickle.load(fp)

	with open ('rewards', 'rb') as fp:
		rewards = pickle.load(fp)

	with open ('psd_rewards', 'rb') as fp:
		psd_rewards = pickle.load(fp)

	parser.add_argument("--exp_idx", help="Index of experiment")
	parser.add_argument('--size', default=10000, type=int, help='bandit size')
	parser.add_argument('--dataset', default='mnist', metavar='DATASET')
	parser.add_argument('--shuffle', type=bool, default=1, metavar='1 / 0', help='shuffle the data set or not')
	parser.add_argument('--seed', type=int, default=0, help='random seed for shuffle, 0 for None')
	parser.add_argument('--nu', type=float, default=1, metavar='v', help='nu for control variance')
	parser.add_argument('--lamdba', type=float, default=0.001, metavar='l', help='lambda for regularzation')
	parser.add_argument('--hidden', type=int, default=100, help='network hidden size')

	args = parser.parse_args()
	l = NeuralLinearUCB(20, args.lamdba, args.nu, args.hidden)
	ucb_info = '_{:.3e}_{:.3e}'.format(args.lamdba, args.nu)

	regrets = []
	summ = 0
	for t in range(10000):
		context, rwd, psd_rwd = contexts[t], rewards[t], psd_rewards[t]
		arm_select = l.select(context)
		reg = np.max(psd_rwd) - psd_rwd[arm_select]
		summ+=reg
		l.update_model(context, arm_select, rwd)
		if t<2000:
			loss = l.train(context, arm_select, rwd[arm_select])
		else:
			if t%100 == 0:
				loss = l.train(context, arm_select, rwd[arm_select])
		regrets.append(summ)
		if t % 100 == 0:
			print('{}: {:.3f}, {:.3e}'.format(t, summ, loss))
	   
	path = "out/logs/demo/linear_neural_greedy" + str(args.exp_idx)
	fr = open(path,'w')
	for i in regrets:
		fr.write(str(i))
		fr.write("\n")
	fr.close()