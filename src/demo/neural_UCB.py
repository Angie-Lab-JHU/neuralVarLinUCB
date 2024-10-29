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
		out_mu, ucb = [], []
		for fx in mu:
			self.func.zero_grad()
			fx.backward(retain_graph=True)
			g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
			g_list.append(g)
			sigma2 = self.lamdba * self.nu * g * g / self.U
			sigma = torch.sqrt(torch.sum(sigma2))

			sample_r = fx.item() + sigma.item()
			ucb.append(sigma.item())
			out_mu.append(fx.item())

			sampled.append(sample_r)
			ave_sigma += sigma.item()
			ave_rew += sample_r
		arm = np.argmax(sampled)
		self.U += g_list[arm] * g_list[arm]
		return arm, g_list[arm].norm().item(), ave_sigma, ave_rew, [out_mu, ucb]

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

	def validate_model(self, exp_idx, t_idx, contexts_val, psd_rewards_val):
		outputs, labels = [], []
		for t in range(10000):
			context, psd_rwd = contexts_val[t], psd_rewards_val[t]
			arm_select, nrm, sig, ave_rwd, output = self.select(context)
			outputs.append(output)
			labels.append(psd_rwd)

		with open('neural_UCB/outputs_val_' + str(exp_idx) + str(t_idx) + '.pkl', 'wb') as f:
			pickle.dump(outputs, f)

		with open('neural_UCB/labels_val_' + str(exp_idx) + str(t_idx) + '.pkl', 'wb') as f:
			pickle.dump(labels, f)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	with open ('contexts', 'rb') as fp:
		contexts = pickle.load(fp)

	with open ('rewards', 'rb') as fp:
		rewards = pickle.load(fp)
	
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
	l = NeuralUCBDiag(20, args.lamdba, args.nu, args.hidden)
	ucb_info = '_{:.3e}_{:.3e}'.format(args.lamdba, args.nu)

	regrets = []
	summ = 0
	outputs, labels = [], []
	for t in range(10000):
		context, rwd, psd_rwd = contexts[t], rewards[t], psd_rewards[t]
		arm_select, nrm, sig, ave_rwd, output = l.select(context)
		reg = np.max(psd_rwd) - psd_rwd[arm_select]
		outputs.append(output)
		labels.append(psd_rwd)
		summ+=reg
		if t<2000:
			loss = l.train(context[arm_select], rwd[arm_select])
		else:
			if t%100 == 0:
				loss = l.train(context[arm_select], rwd[arm_select])
		regrets.append(summ)
		if t % 100 == 0:
			print('{}: {:.3f}, {:.3e}, {:.3e}, {:.3e}, {:.3e}'.format(t, summ, loss, nrm, sig, ave_rwd))
		
		if t in [0, 1000, 2000, 5000, 7500, 9999]:
			l.validate_model(args.exp_idx, t, contexts_val, psd_rewards_val)
	   
	path = "out/logs/demo/neural_UCB" + str(args.exp_idx)
	fr = open(path,'w')
	for i in regrets:
		fr.write(str(i))
		fr.write("\n")
	fr.close()

	with open('neural_UCB/neural_UCB_outputs' + str(args.exp_idx) + '.pkl', 'wb') as f:
		pickle.dump(outputs, f)

	with open('neural_UCB/neural_UCB_labels' + str(args.exp_idx) + '.pkl', 'wb') as f:
		pickle.dump(labels, f)