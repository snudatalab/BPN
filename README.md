# Belief Propagation Networks

This is a PyTorch implementation of Belief Propagation Networks.
Refer to the following paper for detailed information.
- Jaemin Yoo, Hyunsik Jeon and U Kang, Belief Propagation Network for Hard 
Inductive Semi-Supervised Learning (IJCAI 2019)

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

## Requirements

The following Python packages are required.
All codes are written by Python 3.5.

- torch
- numpy
- scipy
- click

## Demo

You can run a demo script `demo.sh` that reproduces the experimental results in
the paper by the following command.
Three of the four datasets are included except the `Amazon` dataset.
You can change the hyperparameters by modifying `main.py`.
```
bash demo.sh
```

## Data

Preprocessed data are included in the `data` directory.
Functions for loading the data are based on the implementation of a [graph
convolutional network (GCN)](https://github.com/tkipf/gcn).
You can use your own data if it is a graph, each node contains a feature vector,
and at least a few labels have been observed. 

## Citation

Please cite our paper if you use this code in your own work:
```
@inproceedings{YooJK19,
  author    = {Jaemin Yoo and
               Hyunsik Jeon and
               U Kang},
  title     = {Belief Propagation Network for Hard Inductive Semi-Supervised Learning},
  booktitle = {International Joint Conference on Artificial Intelligence (IJCAI)},
  year      = {2019},
}
```
