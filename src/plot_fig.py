import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.figure(constrained_layout=True)
plt.rcParams.update({'font.size': 15})

def get_regret(filename):
	# Using readlines()
	file1 = open(filename, 'r')
	Lines = file1.readlines()

	regret_list = []
	count = 0
	# Strips the newline character
	for line in Lines:
		count += 1
		regret_list.append(float(line.strip()))
	
	return regret_list

linear_regret = get_regret('out/logs/demo/demoo_0/linear_UCB1')
linear_regret2 = get_regret('out/logs/demo/demoo_0/linear_UCB2')
linear_regret3 = get_regret('out/logs/demo/demoo_0/linear_UCB3')
linear_regret4 = get_regret('out/logs/demo/demoo_0/linear_UCB4')
linear_regret5 = get_regret('out/logs/demo/demoo_0/linear_UCB5')
linear_regret = [linear_regret, linear_regret2, linear_regret3, linear_regret4, linear_regret5]

neural_regret = get_regret('out/logs/demo/demoo_0/neural_UCB1')
neural_regret2 = get_regret('out/logs/demo/demoo_0/neural_UCB2')
neural_regret3 = get_regret('out/logs/demo/demoo_0/neural_UCB3')
neural_regret4 = get_regret('out/logs/demo/demoo_0/neural_UCB4')
neural_regret5 = get_regret('out/logs/demo/demoo_0/neural_UCB5')
neural_regret = [neural_regret, neural_regret2, neural_regret3, neural_regret4, neural_regret5]

linear_neural_greedy_regret = get_regret('out/logs/demo/demoo_0/linear_neural_greedy1')
linear_neural_greedy_regret2 = get_regret('out/logs/demo/demoo_0/linear_neural_greedy2')
linear_neural_greedy_regret3 = get_regret('out/logs/demo/demoo_0/linear_neural_greedy3')
linear_neural_greedy_regret4 = get_regret('out/logs/demo/demoo_0/linear_neural_greedy4')
linear_neural_greedy_regret5 = get_regret('out/logs/demo/demoo_0/linear_neural_greedy5')
linear_neural_greedy_regret = [linear_neural_greedy_regret, linear_neural_greedy_regret2, linear_neural_greedy_regret3,
	linear_neural_greedy_regret4, linear_neural_greedy_regret5]

linear_neural_regret = get_regret('out/logs/demo/demoo_0/linear_neural_UCB1')
linear_neural_regret2 = get_regret('out/logs/demo/demoo_0/linear_neural_UCB2')
linear_neural_regret3 = get_regret('out/logs/demo/demoo_0/linear_neural_UCB3')
linear_neural_regret4 = get_regret('out/logs/demo/demoo_0/linear_neural_UCB4')
linear_neural_regret5 = get_regret('out/logs/demo/demoo_0/linear_neural_UCB5')
linear_neural_regret = [linear_neural_regret, linear_neural_regret2, linear_neural_regret3,
	linear_neural_regret4, linear_neural_regret5]

ours_regret_log = get_regret('out/logs/demo/demoo_0/neural_MLE1')
ours_regret_log2 = get_regret('out/logs/demo/demoo_0/neural_MLE2')
ours_regret_log3 = get_regret('out/logs/demo/demoo_0/neural_MLE3')
ours_regret_log4 = get_regret('out/logs/demo/demoo_0/neural_MLE4')
ours_regret_log5 = get_regret('out/logs/demo/demoo_0/neural_MLE5')
ours_regret_log = [ours_regret_log, ours_regret_log2, ours_regret_log3, ours_regret_log4, ours_regret_log5]

def plot_by_normal(value, label, color = None):
	mean = np.mean(np.array(value), axis = 0)
	std = np.std(np.array(value), axis = 0)
	if color is not None:
		plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.3, color = color)
		plt.plot(mean, label = label, color = color)
	else:
		plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.3)
		plt.plot(mean, label = label)

plot_by_normal(linear_regret, "LinUCB")
plot_by_normal(neural_regret, "NeuralUCB")
plot_by_normal(linear_neural_greedy_regret, "Neural-LinGreedy")
plot_by_normal(linear_neural_regret, "Neural-LinUCB")
plot_by_normal(ours_regret_log, 'Neural-$\sigma^2$-LinUCB', "blue")

plt.xlabel("Steps")
plt.ylabel("Cumulative regret")
plt.ylim(0, 3000)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("out/demo.pdf")