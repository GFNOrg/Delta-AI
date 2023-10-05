import torch
import torch.nn as nn
import torch.nn.functional as F

class BernoulliDist(nn.BCEWithLogitsLoss):
    def __init__(self, perturb_dim, reduction='none'):
        super().__init__(reduction=reduction)
        self.perturb_dim = perturb_dim
    def logprob(self, input, targets):
        return - (super().forward(input[:, 0], (targets + 1) / 2))
    def sample(self, logits, temp=1):
        return (2 * torch.bernoulli(torch.sigmoid(logits[:, 0] / temp)) - 1)
    def perturb(self, V, ilist):
        # Gather V at ilist and flip its sign
        return torch.where(F.one_hot(ilist, num_classes=self.perturb_dim) > 0, -V, V)
