"""Imbalance-aware evidential objective."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def effective_number_weights(counts: torch.Tensor, beta: float = .9999) -> torch.Tensor:
    counts = counts.float().clamp_min(1)
    weights = (1 - beta) / (1 - beta ** counts)
    return weights / weights.mean()


def dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    k = alpha.shape[1]
    beta = torch.ones((1, k), device=alpha.device)
    sum_a, sum_b = alpha.sum(1, keepdim=True), beta.sum(1, keepdim=True)
    return (torch.lgamma(sum_a) - torch.lgamma(alpha).sum(1, keepdim=True)
            - torch.lgamma(sum_b) + torch.lgamma(beta).sum(1, keepdim=True)
            + ((alpha-beta) * (torch.digamma(alpha)-torch.digamma(sum_a))).sum(1, keepdim=True)).mean()


def focal_evidential_loss(output: dict[str, torch.Tensor], target: torch.Tensor,
                           class_weights: torch.Tensor, gamma: float = 2., kl_weight: float = 1e-3,
                           domain_target: torch.Tensor | None = None, domain_weight: float = .1) -> dict[str, torch.Tensor]:
    logp = output["logits"]
    p = logp.exp()
    pt = p.gather(1, target[:, None]).squeeze(1)
    ce = F.nll_loss(logp, target, weight=class_weights, reduction="none")
    focal = ((1 - pt).clamp_min(0) ** gamma * ce).mean()
    annealed_alpha = torch.ones_like(output["alpha"])
    annealed_alpha.scatter_(1, target[:, None], output["alpha"].gather(1, target[:, None]))
    kl = dirichlet_kl_to_uniform(annealed_alpha)
    domain = torch.zeros((), device=target.device)
    if domain_target is not None and output["domain_logits"].shape[1] > 1:
        domain = F.cross_entropy(output["domain_logits"], domain_target)
    total = focal + kl_weight * kl + domain_weight * domain
    return {"loss": total, "classification": focal, "evidence_kl": kl, "domain": domain}
