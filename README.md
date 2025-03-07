# [AISTATS 2025] Near-Polynomially Competitive Active Logistic Regression
Code repository for the paper:

> <a href="https://joeyandbluewhale.github.io/">Yihan Zhou</a>, <a href="https://www.cs.utexas.edu/~ecprice/">Eric Price</a>, and <a href="https://trung6.github.io/">Trung Nguyen</a>. <em>Near-Polynomially Competitive Active Logistic Regression</em>. In Proceedings of the 28th International Conference on Artificial Intelligence and Statistics, 2025.

## Datasets
<ol>
<li>Synthetic dataset consists of points drawn uniformly from a 100-dim hypercube.</li>
<li>Musk (Version 2) (sourced from <a href="https://archive.ics.uci.edu/dataset/75/musk+version+2">here</a>)</li>
</ol>

## Dependencies
- TensorFlow
- TensorFlow Probability
- ucimlrepo

## Running code
Our code is tested with Python 3.11, Tensorflow 2.15, Tensorflow Probability 0.23, CUDA 12.4. We run our query algorithm on a NVIDIA A40 GPU.
- To obtain datasets, run
```python data.py```

- To obtain results for the ACED method, we refer readers to the official implementation at https://github.com/jifanz/ACED

- To query passively, run
```
python passive.py
```

- To query using our algorithm, run
`python ours.py --data <dataset_name> --nque <number of queries>`

- To query using the leverage score sampling (lss) algorithm, run
`python lss.py`

- To obtain results for passive, lss, and our method, run
`python logit_regress.py`

- To produce classification accuracy plots, run
```
python plot.py
```

The code for leverage score sampling has been sourced from <a href="https://github.com/AgnivaC/SubsampledLogisticRegression">here</a>.

### Citation

```
@inproceedings{
zhou2025nearpolynomially,
title={Near-Polynomially Competitive Active Logistic Regression},
author={Yihan Zhou and Eric Price and Trung Nguyen},
booktitle={The 28th International Conference on Artificial Intelligence and Statistics},
year={2025},
url={https://openreview.net/forum?id=BVQ8rIFuYa}
}
```


</br>
Please contact <a href="https://trung6.github.io/">Trung Nguyen</a> for questions or comments.