"""
Belief Propagation Network for Hard Inductive Semi-Supervised Learning

Authors:
- Jaemin Yoo (jaeminyoo@snu.ac.kr), Seoul National University
- Hyunsik Jeon (jeon185@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor


def _count_degrees(edges: ndarray, num_nodes: int) -> ndarray:
    """
    Count the node degrees.
    """
    degrees = np.zeros(num_nodes, dtype=int)
    for edge in edges:
        for node in edge:
            degrees[node] += 1
    return degrees


def _set_indices(degrees: ndarray, num_nodes: int) -> ndarray:
    """
    Set an index matrix.
    """
    indices = np.zeros(num_nodes, dtype=int)
    for i in range(1, num_nodes):
        indices[i] = indices[i - 1] + degrees[i - 1]
    return indices


def _set_src_edges(edges: ndarray, indices: ndarray) -> ndarray:
    """
    Set an edge matrix of source nodes.
    """
    counts = np.zeros(indices.shape[0], dtype=int)
    src_nodes = np.zeros(edges.shape[0] * 2, dtype=int)
    for node1, node2 in edges:
        src_nodes[indices[node2] + counts[node2]] = node1
        src_nodes[indices[node1] + counts[node1]] = node2
        counts[node2] += 1
        counts[node1] += 1
    return src_nodes


def _set_dst_nodes(degrees: ndarray, num_edges: int) -> ndarray:
    """
    Set an edge matrix of destination nodes.
    """
    dst_nodes = np.zeros(num_edges * 2, dtype=int)
    index = 0
    for node, degree in enumerate(degrees):
        for _ in range(degree):
            dst_nodes[index] = node
            index += 1
    return dst_nodes


def _set_rev_edges(degrees: ndarray,
                   indices: ndarray,
                   edges: ndarray) -> ndarray:
    """
    Set a matrix of reversed edges.
    """
    counts = np.zeros(indices.shape[0], dtype=int)
    rev_edges = np.zeros(edges.shape, dtype=int)
    index = 0
    for dst, degree in enumerate(degrees):
        for _ in range(degree):
            src = edges[index]
            rev_edges[indices[src] + counts[src]] = index
            index += 1
            counts[src] += 1
    return rev_edges


class Graph:
    """
    Class for representing an undirected graph.
    """

    def __init__(self, nodes: ndarray, edges: ndarray):
        """
        Initializer.
        """
        super(Graph, self).__init__()

        num_nodes = nodes.shape[0]
        num_edges = edges.shape[0]

        degrees = _count_degrees(edges, num_nodes)
        indices = _set_indices(degrees, num_nodes)
        src_nodes = _set_src_edges(edges, indices)
        dst_nodes = _set_dst_nodes(degrees, num_edges)
        rev_edges = _set_rev_edges(degrees, indices, src_nodes)

        self.features = torch.tensor(nodes, dtype=torch.float)
        self.src_nodes = torch.tensor(src_nodes, dtype=torch.long)
        self.dst_nodes = torch.tensor(dst_nodes, dtype=torch.long)
        self.rev_edges = torch.tensor(rev_edges, dtype=torch.long)

    def to(self, device: torch.device):
        """
        Move current Tensors into a given device.
        """
        self.features = self.features.to(device)
        self.src_nodes = self.src_nodes.to(device)
        self.dst_nodes = self.dst_nodes.to(device)
        self.rev_edges = self.rev_edges.to(device)
        return self

    def num_nodes(self) -> int:
        """
        Count the number of nodes.
        """
        return self.features.shape[0]

    def num_edges(self) -> int:
        """
        Count the number of edges.
        """
        return self.src_nodes.shape[0] // 2

    def get_features(self) -> Tensor:
        """
        Return the node features.
        """
        return self.features
