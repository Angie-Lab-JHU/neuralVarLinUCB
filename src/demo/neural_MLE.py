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
from torch.utils.data import TensorDataset, DataLoader

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
		self.sigm_bound = 1/dim
		self.sigma = self.sigm_bound

	def select(self, context):
		tensor = torch.from_numpy(context).float().cuda()
		features = self.func(tensor).cpu().detach().numpy()
		ucb = np.array([np.sqrt(np.dot(features[a,:], np.dot(self.A_inv[a], features[a,:].T))) for a in range(self.n_arm)])
		mu = np.array([np.dot(features[a,:], self.theta[a]) for a in range(self.n_arm)])
		arm = np.argmax(mu + 0.02 * ucb)
		return arm, mu[arm], [mu, ucb]

	def train(self, context, arm_select, reward):
		self.context_list.append(torch.from_numpy(context[arm_select].reshape(1, -1)).float())
		self.arm_list.append(arm_select)
		self.reward.append(reward)
		optimizer = optim.SGD(self.func.parameters(), lr=1e-2, weight_decay=self.lamdba)
		train_set = []
		for idx in range(len(self.context_list)):
			train_set.append((self.context_list[idx], self.arm_list[idx], self.reward[idx]))
		
		ite = 0
		
		tot_loss = 0
		while True:
			batch_loss = 0
			train_loader = DataLoader(train_set, batch_size = 64, shuffle = True)
			for batch_idx, (samples, arms, labels) in enumerate(train_loader):
				samples = samples.reshape(samples.shape[0] * samples.shape[1], samples.shape[2]).float().cuda()
				labels = labels.reshape(labels.shape[0], 1).cuda()
				optimizer.zero_grad()
				features = self.func(samples.cuda())
				mu = (features * torch.from_numpy(self.theta[arms]).float().cuda()).sum(dim=1, keepdims=True)
				loss = torch.mean((mu - labels)**2)
				loss.backward()
				optimizer.step()
				batch_loss += loss.item()
				tot_loss += loss.item()
				ite += 1
				if ite >= 1000:
					return tot_loss / 1000

	def update_model(self, context, arm_select, reward, mu, a_low, b_up):
		tensor = torch.from_numpy(context).float().cuda()
		context = self.func(tensor).cpu().detach().numpy()
		self.theta = np.array([np.matmul(self.A_inv[a], self.b[a]) for a in range(self.n_arm)])

		self.sigma = (b_up - mu) * (mu - a_low)
		
		self.sigma = max(self.sigma, self.sigm_bound)
		self.b[arm_select] += (context[arm_select] * reward[arm_select])/self.sigma
		self.A_inv[arm_select] = inv_sherman_morrison(context[arm_select,:]/np.sqrt(self.sigma),self.A_inv[arm_select])

	def validate_model(self, exp_idx, t_idx, contexts_val, psd_rewards_val):
		outputs, labels = [], []
		for t in range(10000):
			context, psd_rwd = contexts_val[t], psd_rewards_val[t]
			arm_select, mu, output = self.select(context)
			outputs.append(output)
			labels.append(psd_rwd)

		with open('neural_MLE/outputs_val_' + str(exp_idx) + str(t_idx) + '.pkl', 'wb') as f:
			pickle.dump(outputs, f)

		with open('neural_MLE/labels_val_' + str(exp_idx) + str(t_idx) + '.pkl', 'wb') as f:
			pickle.dump(labels, f)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	with open ('contexts', 'rb') as fp:
		contexts = pickle.load(fp)

	with open ('rewards', 'rb') as fp:
		rewards = pickle.load(fp)
	
	b_up, a_low = np.max(rewards), np.min(rewards)

	with open ('psd_rewards', 'rb') as fp:
		psd_rewards = pickle.load(fp)
	
	contexts_val = contexts[10000:]
	contexts = contexts[:10000]
	rewards_val = rewards[10000:]
	rewards = rewards[:10000]
	psd_rewards_val = psd_rewards[10000:]
	psd_rewards = psd_rewards[:10000]

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
	outputs, labels = [], []
	for t in range(10000):
		context, rwd, psd_rwd = contexts[t], rewards[t], psd_rewards[t]
		arm_select, mu, output = l.select(context)
		reg = np.max(psd_rwd) - psd_rwd[arm_select]
		outputs.append(output)
		labels.append(psd_rwd)
		summ+=reg
		l.update_model(context, arm_select, rwd, mu, a_low, b_up)
		if t<2000:
			loss = l.train(context, arm_select, rwd[arm_select])
		else:
			if t%100 == 0:
				loss = l.train(context, arm_select, rwd[arm_select])
		regrets.append(summ)
		if t % 100 == 0:
			print('{}: {:.3f}, {:.3e}'.format(t, summ, loss))

		if t in [0, 1000, 2000, 5000, 7500, 9999]:
			l.validate_model(args.exp_idx, t, contexts_val, psd_rewards_val)
	   
	path = "out/logs/demo/neural_MLE" + str(args.exp_idx)
	fr = open(path,'w')
	for i in regrets:
		fr.write(str(i))
		fr.write("\n")
	fr.close()

	with open('neural_MLE/outputs' + str(args.exp_idx) + '.pkl', 'wb') as f:
		pickle.dump(outputs, f)

	with open('neural_MLE/labels' + str(args.exp_idx) + '.pkl', 'wb') as f:
		pickle.dump(labels, f)