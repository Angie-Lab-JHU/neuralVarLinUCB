for i in {1..5}; do
	python src/linear_UCB.py --exp_idx $i
	python src/neural_UCB.py --exp_idx $i
	python src/linear_neural_greedy.py --exp_idx $i
	python src/linear_neural_UCB.py --exp_idx $i
	python src/neural_MLE.py --exp_idx $i
done