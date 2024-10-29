import numpy as np
import pickle
import itertools

class Bandit_multi:
	def __init__(self):
		self.dim = 20
		self.n_arms = 4
		a = np.random.randn(self.dim)
		a /= np.linalg.norm(a, ord=2)
		A = np.random.randn(self.dim, self.dim)
		self.h = lambda x: 10*(np.dot(x, a))**2
		# self.h = lambda x: np.matmul(np.matmul(np.matmul(x, A), A), x)
		# self.h = lambda x: np.cos(3*np.dot(x, a))
		self.size = 20000
		self.noise_var= np.random.uniform(0, 1, self.size)
		self.noise_std = np.sqrt(self.noise_var)

	def step(self, t):
		x = np.random.randn(self.n_arms, self.dim)
		x /= np.repeat(np.linalg.norm(x, axis=-1, ord=2), self.dim).reshape(self.n_arms, self.dim)
		
		rwd, psd_rwd = [], []
		for k in itertools.product(range(self.n_arms)):
			tmp = self.h(x[k])
			psd_rwd.append(tmp)
			rwd.append(tmp + self.noise_std[t] * np.random.randn())
		psd_rwd = np.array(psd_rwd).reshape(self.n_arms)
		rwd = np.array(rwd).reshape(self.n_arms)
		
		return x, rwd, psd_rwd, self.noise_std[t]


b = Bandit_multi()
contexts, noise_std, rewards, psd_rewards = [], [], [], []
for t in range(b.size):
	context, rwd, psd_rwd, std = b.step(t)
	contexts.append(context)
	noise_std.append(std)
	rewards.append(rwd)
	psd_rewards.append(psd_rwd)

with open('contexts', 'wb') as fp:
	pickle.dump(contexts, fp)

with open('noise_std', 'wb') as fp:
	pickle.dump(noise_std, fp)

with open('rewards', 'wb') as fp:
	pickle.dump(rewards, fp)

with open('psd_rewards', 'wb') as fp:
	pickle.dump(psd_rewards, fp)
