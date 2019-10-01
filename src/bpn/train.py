"""
Belief Propagation Network for Hard Inductive Semi-Supervised Learning

Authors:
- Jaemin Yoo (jaeminyoo@snu.ac.kr), Seoul National University
- Hyunsik Jeon (jeon185@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""
import io
from collections import Callable

import numpy as np
import torch
from torch import Tensor

from bpn.models import BPN


def _compute_loss(predicted: Tensor,
                  target: Tensor,
                  loss_func: Callable) -> Tensor:
    """
    Compute the loss of a prediction.
    """
    log_priors = torch.log(predicted)
    loss = loss_func(log_priors, target.detach())
    return loss


def _compute_accuracy(beliefs: Tensor, labels: Tensor) -> float:
    """
    Compute the accuracy of a prediction.
    """
    y_pred = beliefs.argmax(dim=1)
    y_true = labels.argmax(dim=1)
    return torch.sum(y_true == y_pred).item() / y_true.size(0)


def fit(model: BPN,
        labels: torch.Tensor,
        trn_nodes: torch.Tensor,
        val_nodes: torch.Tensor,
        learning_rate: float = 1e-1,
        weight_decay: float = 1e-3,
        coefficient: float = 1.,
        verbose: bool = True,
        num_epochs: int = 5000,
        patience: int = 200):
    """
    Train a BPN model.
    """
    identity = torch.eye(model.num_states, device=model.device)
    trn_labels = identity[labels[trn_nodes]]
    val_labels = identity[labels[val_nodes]]

    oth_nodes = torch.ones(
        model.num_nodes(), dtype=torch.uint8, device=model.device)
    oth_nodes[trn_nodes] = 0

    max_accr = 0.
    curr_step = 0
    min_loss = 0
    saved_model = None

    optimizer = torch.optim.Adam(
        model.parameters(), learning_rate, weight_decay=weight_decay)
    loss_func = torch.nn.KLDivLoss(reduction='batchmean')

    if verbose:
        print('Epoch\tClsLoss\tIndLoss\tTrnAcc\tValAcc')

    tmp_result = []
    for epoch in range(num_epochs + 1):
        model.train()
        priors, beliefs = model()
        trn_beliefs = beliefs[trn_nodes]
        oth_beliefs = beliefs[oth_nodes]
        oth_priors = priors[oth_nodes]

        model.eval()
        val_priors = model()[0][val_nodes]

        cls_loss = _compute_loss(trn_beliefs, trn_labels, loss_func)
        ind_loss = _compute_loss(oth_priors, oth_beliefs, loss_func)
        loss = cls_loss * (1 - coefficient) + ind_loss * coefficient

        if epoch > 0:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step(closure=None)

        trn_accr = _compute_accuracy(trn_beliefs, trn_labels)
        val_accr = _compute_accuracy(val_priors, val_labels)

        if val_accr > max_accr or (val_accr == max_accr and ind_loss < min_loss):
            saved_model = io.BytesIO()
            torch.save(model.state_dict(), saved_model)
            max_accr = val_accr
            min_loss = ind_loss
            curr_step = 0
        else:
            curr_step += 1
            if curr_step == patience:
                break

        if verbose:
            tmp_result.append(
                (cls_loss.item(), ind_loss.item(), trn_accr, val_accr))

            if (epoch + 1) % 10 == 0:
                print('{}\t{:.8f}\t{:.8f}\t{:.4f}\t{:.4f}'.format(
                    epoch + 1, *np.average(tmp_result, axis=0)))
                tmp_result = []

    if verbose:
        print()

    saved_model.seek(0)
    model.load_state_dict(torch.load(saved_model))
