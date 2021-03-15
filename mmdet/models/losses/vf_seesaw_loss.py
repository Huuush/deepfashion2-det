import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .varifocal_loss import VarifocalLoss


def seesaw_loss(pred,
                target,
                class_counts,
                eps=1.0e-6,
                p: float = 0.8,
                q: float = 2,
                num_labels=13):
    target = F.one_hot(target, num_labels)

    # Mitigation Factor
    if class_counts is None:
        class_counts = (target.sum(axis=0) + 1).float()  # to prevent devided by zero.
    else:
        class_counts += target.sum(axis=0)

    m_conditions = class_counts[:, None] > class_counts[None, :]
    m_trues = (class_counts[None, :] / class_counts[:, None]) ** p
    m_falses = torch.ones(len(class_counts), len(class_counts)).to(target.device)
    m = torch.where(m_conditions, m_trues, m_falses)  # [num_labels, num_labels]

    # Compensation Factor
    # only error sample need to compute Compensation Factor
    probility = F.softmax(pred, dim=-1)
    c_condition = probility / (probility * target).sum(dim=-1)[:, None]  # [B, num_labels]
    c_condition = torch.stack([c_condition] * target.shape[-1], dim=1)  # [B, N, N]
    c_condition = c_condition * target[:, :, None]  # [B, N, N]
    false = torch.ones(c_condition.shape).to(target.device)  # [B, N, N]
    c = torch.where(c_condition > 1, c_condition ** q, false)  # [B, N, N]

    # Sij = Mij * Cij
    s = m[None, :, :] * c
    # softmax trick to prevent overflow (like logsumexp trick)
    max_element, _ = pred.max(axis=-1)
    pred = pred - max_element[:, None]  # to prevent overflow
    numerator = torch.exp(pred)
    denominator = (
        (1 - target)[:, None, :]
        * s[None, :, :]
        * torch.exp(pred)[:, None, :]).sum(axis=-1) \
        + torch.exp(pred)

    sigma = numerator / (denominator + eps)
    loss = (- target * torch.log(sigma + eps)).sum(-1)
    return loss.mean()


@LOSSES.register_module()
class Vf_seesaw_loss(nn.Module):
    def __init__(self,
                 loss_weight=0.1):
        """
        a loss with vfloss and seesaw loss.
            loss_weight: how to add two loss
        """
        super(Vf_seesaw_loss, self).__init__()
        self.loss_weight = loss_weight
        self.vf_loss = VarifocalLoss()

    def forward(self,
                pred,
                target,
                weight=None):

        loss1 = self.vf_loss(pred, target)
        loss2 = self.loss_weight * seesaw_loss(
            pred,
            target,
            weight
        )

        loss = loss1 + loss2

        return loss


# class DistibutionAgnosticSeesawLossWithLogits(nn.Module):
#     """
#     This is unofficial implementation for Seesaw loss,
#     which is proposed in the techinical report for LVIS workshop at ECCV 2020.
#     For more detail, please refer https://arxiv.org/pdf/2008.10032.pdf.
#     Args:
#     p: Parameter for Mitigation Factor,
#        Set to 0.8 for default following the paper.
#     q: Parameter for Compensation Factor
#        Set to 2 for default following the paper.
#     num_labels: Class nums
#     """
#
#     def __init__(self, p: float = 0.8, q: float = 2, num_labels=13):
#         super().__init__()
#         self.eps = 1.0e-6
#         self.p = p
#         self.q = q
#         self.class_counts = None
#         self.num_labels = num_labels
#
#     def forward(self, logits, targets):
#         targets = F.one_hot(targets, self.num_labels)
#
#         # Mitigation Factor
#         if self.class_counts is None:
#             self.class_counts = (targets.sum(axis=0) + 1).float()  # to prevent devided by zero.
#         else:
#             self.class_counts += targets.sum(axis=0)
#
#         m_conditions = self.class_counts[:, None] > self.class_counts[None, :]
#         m_trues = (self.class_counts[None, :] / self.class_counts[:, None]) ** self.p
#         m_falses = torch.ones(len(self.class_counts), len(self.class_counts)).to(targets.device)
#         m = torch.where(m_conditions, m_trues, m_falses)  # [num_labels, num_labels]
#
#         # Compensation Factor
#         # only error sample need to compute Compensation Factor
#         probility = F.softmax(logits, dim=-1)
#         c_condition = probility / (probility * targets).sum(dim=-1)[:, None]  # [B, num_labels]
#         c_condition = torch.stack([c_condition] * targets.shape[-1], dim=1)  # [B, N, N]
#         c_condition = c_condition * targets[:, :, None]  # [B, N, N]
#         false = torch.ones(c_condition.shape).to(targets.device)  # [B, N, N]
#         c = torch.where(c_condition > 1, c_condition ** self.q, false)  # [B, N, N]
#
#         # Sij = Mij * Cij
#         s = m[None, :, :] * c
#         # softmax trick to prevent overflow (like logsumexp trick)
#         max_element, _ = logits.max(axis=-1)
#         logits = logits - max_element[:, None]  # to prevent overflow
#         numerator = torch.exp(logits)
#         denominator = (
#                               (1 - targets)[:, None, :]
#                               * s[None, :, :]
#                               * torch.exp(logits)[:, None, :]).sum(axis=-1) \
#                       + torch.exp(logits)
#
#         sigma = numerator / (denominator + self.eps)
#         loss = (- targets * torch.log(sigma + self.eps)).sum(-1)
#         return loss.mean()
