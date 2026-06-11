import torch
import torch.nn.functional as F
from torch import nn


class InfoNCE(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, features):
        features = F.normalize(features, dim=1)
        batch_size = features.size(0) // 2

        a = features[:batch_size, :]
        b = features[batch_size:, :]

        cos_aa = a @ a.T / self.temperature
        cos_bb = b @ b.T / self.temperature
        cos_ab = a @ b.T / self.temperature

        log_numerator = cos_ab.trace() / batch_size

        self_mask = torch.eye(batch_size, dtype=bool, device=cos_aa.device)
        cos_aa.masked_fill_(self_mask, float('-inf'))
        cos_bb.masked_fill_(self_mask, float('-inf'))
        logsumexp_1 = torch.hstack((cos_ab.T, cos_bb)).logsumexp(dim=1).mean()
        logsumexp_2 = torch.hstack((cos_aa, cos_ab)).logsumexp(dim=1).mean()
        log_denominator = (logsumexp_1 + logsumexp_2) / 2

        return -(log_numerator - log_denominator)


class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, features):
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0] // 2

        similarity = features @ features.T / self.temperature

        # mask self-similarity
        self_mask = torch.eye(2 * batch_size, device=features.device).bool()
        similarity = similarity.masked_fill(self_mask, float('-inf'))

        # positive pairs
        targets = torch.arange(batch_size, device=features.device)
        targets = torch.cat([targets + batch_size, targets], dim=0)

        loss = F.cross_entropy(similarity, targets)
        return loss


if __name__ == '__main__':
    pass
