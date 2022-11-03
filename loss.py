import torch
import torch.nn as nn
import torch.nn.functional as F

# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction,
        )


_criterion_entrypoints = {
    "cross_entropy": nn.CrossEntropyLoss,
    "focal": FocalLoss,
}


def is_criterion(loss_name):
    return loss_name in _criterion_entrypoints


def create_criterion(loss_name, **kwargs):
    if is_criterion(loss_name):
        create_fn = _criterion_entrypoints[loss_name]

        if (
            loss_name == "cross_entropy" or loss_name == "focal"
        ) and "classes" in kwargs.keys():
            del kwargs["classes"]

        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError("Unknown loss (%s)" % loss_name)
    return criterion
