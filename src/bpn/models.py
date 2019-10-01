"""
Belief Propagation Network for Hard Inductive Semi-Supervised Learning

Authors:
- Jaemin Yoo (jaeminyoo@snu.ac.kr), Seoul National University
- Hyunsik Jeon (jeon185@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""
import torch
from torch import Tensor
from torch import nn
from torch.nn import Module
from torch.nn import functional as func

from bpn.graph import Graph

EPSILON = 1e-10


class MLP(Module):
    """
    Multi-layer perceptron as a classifier.
    """

    def __init__(self,
                 num_attributes: int,
                 num_states: int,
                 layers: tuple,
                 bias: bool = False,
                 dropout: float = 0.,
                 activation: str = 'tanh'):
        """
        Initializer.
        """
        super().__init__()

        self.activation = activation
        self.layers_linear = nn.ModuleList()
        self.layers_dropout = nn.ModuleList()

        num_units = num_attributes
        for h in layers:
            self.layers_linear.append(nn.Linear(num_units, h, bias))
            self.layers_dropout.append(nn.Dropout(dropout))
            num_units = h

        self.last_layer = nn.Linear(num_units, num_states, bias=False)

    def _apply_activation(self, tensor: Tensor) -> Module:
        """
        Apply an activation function after a linear layer.
        """
        if self.activation == 'relu':
            return func.relu(tensor)
        elif self.activation == 'tanh':
            return torch.tanh(tensor)
        else:
            raise ValueError(self.activation)

    def forward(self, features: torch.Tensor) -> Module:
        """
        Forward function.
        """
        x = features
        for lin, do in zip(self.layers_linear, self.layers_dropout):
            x = lin(x)
            x = self._apply_activation(x)
            x = do(x)
        x = self.last_layer(x)
        return func.softmax(x, dim=1)


class BPN(Module):
    """
    Belief propagation network.
    """

    def __init__(self,
                 graph: Graph,
                 classifier: Module,
                 num_states: int,
                 device: torch.device,
                 epsilon: float = 1.0,
                 diffusion: int = 1):
        """
        Initializer.
        """
        super(BPN, self).__init__()

        self.num_states = num_states
        self.diffusion = diffusion
        self.epsilon = epsilon
        self.device = device

        self.softmax = nn.Softmax(dim=1)
        self.classifier = classifier
        self.graph = graph
        self.num_edges = self.graph.num_edges()
        self.features = self.graph.get_features()
        self.potential = torch.exp(
            torch.eye(self.num_states, device=device) * self.epsilon)

    def _init_messages(self) -> Tensor:
        """
        Initialize (or create) a message matrix.
        """
        size = (self.num_edges * 2, self.num_states)
        return torch.ones(size, device=self.device) / self.num_states

    def _update_messages(self, messages: Tensor, beliefs: Tensor) -> Tensor:
        """
        Update the message matrix with using beliefs.
        """
        new_beliefs = beliefs[self.graph.src_nodes]
        rev_messages = messages[self.graph.rev_edges]
        new_msgs = torch.mm(new_beliefs / rev_messages, self.potential)
        new_msgs = new_msgs / new_msgs.sum(dim=1, keepdim=True)
        return new_msgs

    def _compute_beliefs(self, priors: Tensor, messages: Tensor) -> Tensor:
        """
        Compute new beliefs based on the current messages.
        """
        beliefs = torch.log(torch.clamp(priors, min=EPSILON))
        log_msgs = torch.log(torch.clamp(messages, min=EPSILON))
        beliefs.index_add_(0, self.graph.dst_nodes, log_msgs)
        return self.softmax(beliefs)

    def propagate(self, priors: Tensor):
        """
        Propagate the priors produced from the classifier.
        """
        beliefs = priors
        messages = self._init_messages()
        for _ in range(self.diffusion):
            messages = self._update_messages(messages, beliefs)
            beliefs = self._compute_beliefs(priors, messages)
        return beliefs

    def forward(self) -> (Tensor, Tensor):
        """
        Run the forward propagation of this model.
        """
        priors = self.classifier(self.features)
        beliefs = self.propagate(priors)
        return priors, beliefs

    def num_nodes(self) -> int:
        """
        Count the number of nodes in the current graph.
        """
        return self.graph.num_nodes()
