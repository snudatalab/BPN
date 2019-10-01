"""
Belief Propagation Network for Hard Inductive Semi-Supervised Learning

Authors:
- Jaemin Yoo (jaeminyoo@snu.ac.kr), Seoul National University
- Hyunsik Jeon (jeon185@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""
import pickle as pkl
import time

import click
import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor
from torch.nn import Module

import bpn.train as train
from bpn.graph import Graph
from bpn.models import BPN, MLP

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(dataset: str, path: str, num_validations: int = 500) -> tuple:
    """
    Load graph data.

    This function is based on the implementation of GCN in the following link:
    https://github.com/tkipf/gcn
    """

    def load_index(filename: str):
        return [int(e) for e in open(filename).readlines()]

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        name = "{}/ind.{}.{}".format(path, dataset, names[i])
        with open(name, 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    name = "{}/ind.{}.test.index".format(path, dataset)
    test_idx_reorder = load_index(name)
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder),
                                    max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = np.array(sp.vstack((allx, tx)).todense())
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.argmax(np.vstack((ally, ty)), axis=1)
    labels[test_idx_reorder] = labels[test_idx_range]

    idx_test = test_idx_range.tolist()
    idx_train = np.arange(len(y))
    idx_val = np.arange(len(y), len(y) + num_validations)

    # Change the indices of test nodes as the last 1000 numbers.

    idx_count1 = 0
    idx_count2 = labels.shape[0] - 1000
    idx_test_set = set(idx_test)
    idx_map = {}
    for n in range(labels.shape[0]):
        if n not in idx_test_set:
            idx_map[n] = idx_count1
            idx_count1 += 1
        else:
            idx_map[n] = idx_count2
            idx_count2 += 1
    idx_reordered = [v[0] for v in sorted(idx_map.items(), key=lambda v: v[1])]

    new_graph = {}
    for node, edges in graph.items():
        new_graph[idx_map[node]] = [idx_map[n] for n in edges]
    features = features[idx_reordered, :]
    labels = labels[idx_reordered]
    idx_test = np.arange(labels.shape[0] - 1000, labels.shape[0])

    return new_graph, features, labels, idx_train, idx_val, idx_test


def reset_parameters(x: Module):
    """
    Reset the parameters of linear layers in a model.
    """
    if isinstance(x, torch.nn.Linear):
        x.reset_parameters()


def select_seeds(seed: int, size: int):
    """
    Select randomly a list of random seeds.
    """
    np.random.seed(seed)
    return np.random.randint(1000000, size=size)


def set_seed(seed: int):
    """
    Set a random seed for PyTorch and numpy.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def compute_accuracy(labels: Tensor, preds: Tensor, nodes: Tensor) -> float:
    """
    Compute the accuracy of a prediction.
    """
    corrects = torch.sum(labels[nodes] == preds[nodes].argmax(dim=1))
    return corrects.item() / nodes.size(0)


@click.command()
@click.option('--dataset', type=str, default='citeseer')
@click.option('--verbose', type=bool, is_flag=True)
def main(dataset: str, verbose: bool):
    """
    Main script for training and evaluating a belief propagation network.
    """
    if dataset == 'pubmed':
        epsilon = 1.0
        decay = 2e-3
        coefficient = 0.5
        lrn_rate = 0.001
        num_states = 3
    elif dataset == 'cora':
        epsilon = 0.05
        decay = 2e-4
        coefficient = 0.9
        lrn_rate = 0.001
        num_states = 7
    elif dataset == 'citeseer':
        epsilon = 0.01
        decay = 2e-4
        coefficient = 0.9
        lrn_rate = 0.01
        num_states = 6
    elif dataset == 'amazon':  # not included in the repo.
        epsilon = 1.0
        decay = 2e-4
        coefficient = 0.5
        lrn_rate = 0.01
        num_states = 3
    else:
        raise ValueError(dataset)

    values = load_data(dataset, path='../data')
    raw_edges, raw_nodes, labels, trn_nodes, val_nodes, test_nodes = tuple(values)

    index = np.ones(raw_nodes.shape[0], dtype=bool)
    index[test_nodes] = 0
    nodes = raw_nodes[index]
    features = torch.tensor(raw_nodes, dtype=torch.float32, device=DEVICE)

    edges = []
    for u, neighbors in raw_edges.items():
        for v in sorted(set(neighbors)):
            if u < v and u not in test_nodes and v not in test_nodes:
                edges.append((u, v))
    edges = np.array(edges)

    graph = Graph(nodes, edges).to(DEVICE)

    classifier = MLP(nodes.shape[1], num_states,
                     bias=False,
                     activation='tanh',
                     layers=(32,),
                     dropout=0.5).to(DEVICE)

    model = BPN(graph, classifier, num_states, DEVICE, epsilon, diffusion=1)

    labels = torch.tensor(labels, device=DEVICE)
    trn_nodes = torch.tensor(trn_nodes, device=DEVICE)
    val_nodes = torch.tensor(val_nodes, device=DEVICE)
    test_nodes = torch.tensor(test_nodes, device=DEVICE)

    seed = 2018
    num_repeats = 10

    epochs, val_accs, test_accs, times = [], [], [], []
    for local_seed in select_seeds(seed, num_repeats):
        set_seed(local_seed)
        model.apply(reset_parameters)

        s_time = time.time()
        train.fit(model, labels, trn_nodes, val_nodes,
                  verbose=verbose,
                  learning_rate=lrn_rate,
                  weight_decay=decay,
                  coefficient=coefficient)
        e_time = time.time()

        classifier.eval()
        preds = classifier(features)

        val_accs.append(compute_accuracy(labels, preds, val_nodes))
        test_accs.append(compute_accuracy(labels, preds, test_nodes))
        times.append(e_time - s_time)

    print('Dataset: {}'.format(dataset))
    print('Validation accuracy (%): {:.1f} ({:.1f})'.format(
        np.average(val_accs) * 100,
        np.std(val_accs) * 100))
    print('Test accuracy (%): {:.1f} ({:.1f})'.format(
        np.average(test_accs) * 100,
        np.std(test_accs) * 100))
    print('Training time (seconds): {:.1f} ({:.1f})'.format(
        np.average(times),
        np.std(times)))


if __name__ == '__main__':
    main()
