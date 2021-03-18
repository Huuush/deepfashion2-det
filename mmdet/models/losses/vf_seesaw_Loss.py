import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


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


def varifocal_loss(pred,
                   target,
                   weight=None,
                   alpha=0.75,
                   gamma=2.0,
                   iou_weighted=True,
                   reduction='mean',
                   avg_factor=None):
    # pred and target should be of the same size
    assert pred.size() == target.size()
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    if iou_weighted:
        focal_weight = target * (target > 0.0).float() + \
                       alpha * (pred_sigmoid - target).abs().pow(gamma) * \
                       (target <= 0.0).float()
    else:
        focal_weight = (target > 0.0).float() + \
                       alpha * (pred_sigmoid - target).abs().pow(gamma) * \
                       (target <= 0.0).float()
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class Varifocal_seesaw_Loss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 alpha=0.75,
                 gamma=2.0,
                 iou_weighted=True,
                 reduction='mean',
                 loss_weight=1.0):
        super(Varifocal_seesaw_Loss, self).__init__()
        assert use_sigmoid is True, \
            'Only sigmoid varifocal loss supported now.'
        assert alpha >= 0.0
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weighted = iou_weighted
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_seesaw = seesaw_loss(pred, target, class_counts=None)
            loss_vf = self.loss_weight * varifocal_loss(
                pred,
                target,
                weight,
                alpha=self.alpha,
                gamma=self.gamma,
                iou_weighted=self.iou_weighted,
                reduction=reduction,
                avg_factor=avg_factor)
            loss_cls = loss_vf + 0.1 * loss_seesaw
        else:
            raise NotImplementedError
        return loss_cls
