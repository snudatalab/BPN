# Belief Propagation Networks

This project is a PyTorch implementation of [Belief Propagation Network for Hard Inductive Semi-Supervised Learning](https://www.ijcai.org/proceedings/2019/0580.pdf), published as a conference proceeding at [IJCAI 2019](http://ijcai19.org/).
This paper proposes a novel approach for hard inductive learning on graph-structured data, where the graph is not given at the test time and thus previous approaches fail with low accuracy.

## License

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

- Jaemin Yoo (jaeminyoo@snu.ac.kr), Seoul National University
- Hyunsik Jeon (jeon185@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

## Abstract

Given graph-structured data, how can we train a robust classifier in a
semi-supervised setting that performs well without neighborhood information?
In this work, we propose belief propagation networks (BPN), a novel approach to
train a deep neural network in a hard inductive setting, where the test data are
given without neighborhood information.
BPN uses a differentiable classifier to compute the prior distributions of nodes,
and then diffuses the priors through the graphical structure, independently from
the prior computation.
This separable structure improves the generalization performance of BPN for
isolated test instances, compared with previous approaches that jointly use the
feature and neighborhood without distinction.
As a result, BPN outperforms state-of-the-art methods in four datasets with an
average margin of 2.4% points in accuracy.

## Prerequisites

- Python 3.5+
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [Scipy](https://scipy.org)
- [Click](https://click.palletsprojects.com/en/7.x/)

## Usage

You can run a demo script `demo.sh` that reproduces the experimental results in
the paper by the following command.
Three of the four datasets are included except the `Amazon` dataset.
You can change the hyperparameters by modifying `main.py`.
```
bash demo.sh
```

## Datasets

Preprocessed data are downloaded from [here](https://github.com/kimiyoung/planetoid) and included in the `data` directory.
Functions for loading the data are based on the implementation of a [graph convolutional network (GCN)](https://github.com/tkipf/gcn).
You can use your own data if it is a graph, each node contains a feature vector, and at least a few labels have been observed. 

| Name | Nodes | Edges | Attributes | Labels | Download |
| :---: | ----: | ----: | ---------: | -----: | :---: |
| Pubmed | 19,717 | 44,324 | 500 | 3 | [Link](https://github.com/kimiyoung/planetoid) |
| Cora | 2,708 | 5,278 | 1,433 | 7 | [Link](https://github.com/kimiyoung/planetoid) |
| Citeseer | 3,327 | 4,552 | 3,703 | 6 | [Link](https://github.com/kimiyoung/planetoid) |
| Amazon | 32,966 | 63,285 | 3,000 | 3 | - |

## Reference

Please cite our paper if you use this code in your own work:
```
@inproceedings{YooJK19,
  author    = {Jaemin Yoo and Hyunsik Jeon and U Kang},
  title     = {Belief Propagation Network for Hard Inductive Semi-Supervised Learning},
  booktitle = {International Joint Conference on Artificial Intelligence (IJCAI)},
  year      = {2019},
}
```
