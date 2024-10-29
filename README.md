# neuralVarLinUCB

## <a name="demo"></a> Quick Demo
Run [this Google Colab](https://colab.research.google.com/drive/1Q6rMOwV6cBQeTQTxwmE2dhjeOq7-xqox?usp=sharing).

## <a name="prepare"></a> To prepare:
### <a name="library">Library</a>
Install prerequisite packages:
```sh
pip install -r requirements.txt
```

### <a name="dataset">Dataset</a>
Generate synthetic dataset:
```sh
python src/demo/sample_dataset.py
```

## <a name="experiments"></a> To run experiments:
### <a name="Synthetic dataset">Synthetic dataset</a>
```sh
python <method_file> --exp_idx=<idx>
```
where the parameters are the following:
- `<method_file>`: file stored the code of method in the `src/demo/` folder. E.g., `<method_file> = src/demo/neural_MLE.py`
- `<idx>`: index of experiment. E.g., `<idx> = 1`

### <a name="Real-world dataset">Real-world dataset</a>
```sh
python <method_file> --exp_idx=<idx>
```
where the parameters are the following:
- `<method_file>`: file stored the code of method in the `src/`. E.g., `<method_file> = src/neural_MLE.py`
- `<idx>`: index of experiment. E.g., `<idx> = 1`

## References
Based on code of:
> [neural_exploration](https://github.com/sauxpa/neural_exploration)\
> github.

> [NeuralUCB](https://github.com/uclaml/neuralucb)\
> github.

> [Supplementary Material](https://openreview.net/forum?id=xnYACQquaGV)\
> openreview.


## License
This source code is released under the Apache-2.0 license, included [here](LICENSE).