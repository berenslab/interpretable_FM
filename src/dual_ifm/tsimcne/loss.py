import torch
import torch.nn.functional as F
from torch import nn


class InfoNCECosine(nn.Module):
    def __init__(
        self, temperature: float = 0.5, reg_coef: float = 0, reg_radius: float = 200
    ):
        super().__init__()
        self.temperature = temperature
        self.reg_coef = reg_coef
        self.reg_radius = reg_radius

    def forward(self, features):
        batch_size = features.size(0) // 2

        a = features[:batch_size]
        b = features[batch_size:]

        # mean deviation from the sphere with radius `reg_radius`
        vecnorms = torch.linalg.vector_norm(features, dim=1)
        target = torch.full_like(vecnorms, self.reg_radius)
        penalty = self.reg_coef * F.mse_loss(vecnorms, target)

        a = F.normalize(a)
        b = F.normalize(b)

        cos_aa = a @ a.T / self.temperature
        cos_bb = b @ b.T / self.temperature
        cos_ab = a @ b.T / self.temperature

        # mean of the diagonal
        tempered_alignment = cos_ab.trace() / batch_size

        # exclude self inner product
        self_mask = torch.eye(batch_size, dtype=bool, device=cos_aa.device)
        cos_aa.masked_fill_(self_mask, float('-inf'))
        cos_bb.masked_fill_(self_mask, float('-inf'))
        logsumexp_1 = torch.hstack((cos_ab.T, cos_bb)).logsumexp(dim=1).mean()
        logsumexp_2 = torch.hstack((cos_aa, cos_ab)).logsumexp(dim=1).mean()
        raw_uniformity = logsumexp_1 + logsumexp_2

        loss = -(tempered_alignment - raw_uniformity / 2) + penalty
        return loss


class InfoNCECauchy(nn.Module):
    def __init__(self, temperature: float = 1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, log_dist=False):
        batch_size = features.size(0) // 2

        a = features[:batch_size]
        b = features[batch_size:]

        dist_aa = torch.cdist(a, a) * self.temperature
        dist_bb = torch.cdist(b, b) * self.temperature
        dist_ab = torch.cdist(a, b) * self.temperature

        sim_aa = 1 / (1 + dist_aa.square())
        sim_bb = 1 / (1 + dist_bb.square())
        sim_ab = 1 / (1 + dist_ab.square())

        tempered_alignment = sim_ab.diagonal().log().mean()

        # exclude self inner product
        self_mask = torch.eye(batch_size, dtype=bool, device=sim_aa.device)
        sim_aa.masked_fill_(self_mask, 0.0)
        sim_bb.masked_fill_(self_mask, 0.0)

        logsumexp_1 = torch.hstack((sim_ab.T, sim_bb)).sum(dim=1).log().mean()
        logsumexp_2 = torch.hstack((sim_aa, sim_ab)).sum(dim=1).log().mean()

        raw_uniformity = (logsumexp_1 + logsumexp_2) / 2

        loss = -(tempered_alignment - raw_uniformity)

        if log_dist:
            dist_pos = dist_ab.diag()
            dist_neg = torch.cat(
                [dist_ab[~self_mask], dist_aa[~self_mask], dist_bb[~self_mask]]
            )
        else:
            dist_pos, dist_neg = 0, 0

        return loss, dist_pos, dist_neg


class InfoNCEGaussian(nn.Module):
    def __init__(self, temperature=1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features):
        batch_size = features.size(0) // 2

        a = features[:batch_size]
        b = features[batch_size:]

        sim_aa = -(torch.cdist(a, a) * self.temperature).square()
        sim_bb = -(torch.cdist(b, b) * self.temperature).square()
        sim_ab = -(torch.cdist(a, b) * self.temperature).square()

        tempered_alignment = sim_ab.trace() / batch_size

        # exclude self inner product
        self_mask = torch.eye(batch_size, dtype=bool, device=sim_aa.device)
        sim_aa.masked_fill_(self_mask, float('-inf'))
        sim_bb.masked_fill_(self_mask, float('-inf'))

        logsumexp_1 = torch.hstack((sim_ab.T, sim_bb)).logsumexp(1).mean()
        logsumexp_2 = torch.hstack((sim_aa, sim_ab)).logsumexp(1).mean()

        raw_uniformity = logsumexp_1 + logsumexp_2

        loss = -(tempered_alignment - raw_uniformity / 2)
        return loss


if __name__ == '__main__':
    pass
