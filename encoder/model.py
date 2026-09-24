"""TrajectoryEncoder: raw [q|v|a|j] trajectory in, one style latent out.

A learnable filterbank (parallel stride-1 dilated convs whose per-band |.|-energy pooling forms a
learned spectral representation) plus a strided temporal branch -> masked global pooling over time
(so any length works) -> a Gaussian latent (mean, log-variance), with an auxiliary head that
regresses the style fingerprint.

The module and attribute names fix the state_dict keys. cuRobo's VaeManifoldCost re-implements
this class and loads checkpoints with strict=True, so do not rename or reorder layers.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean_std(x, m):                                 # x:(B,C,T) m:(B,1,T) -> (mean, std)
    s = m.sum(-1).clamp(min=1.0)
    mean = (x * m).sum(-1) / s
    std = (((x - mean.unsqueeze(-1)) ** 2 * m).sum(-1) / s).clamp(min=1e-8).sqrt()
    return mean, std


def masked_stats(x, m):                                    # masked mean | std | max over time
    mean, std = masked_mean_std(x, m)
    return torch.cat([mean, std, (x + (1 - m) * -1e9).amax(-1)], 1)


class TrajectoryEncoder(nn.Module):
    def __init__(self, ch, d, n_target, emb=96, p=0.2):
        super().__init__()
        specs = [(3, 1), (7, 1), (15, 1), (7, 2), (15, 4), (15, 8)]
        self.fb = nn.ModuleList([nn.Conv1d(ch, 32, k, dilation=dl, padding=dl * (k - 1) // 2)
                                 for k, dl in specs])
        self.fbn = nn.ModuleList([nn.BatchNorm1d(32) for _ in specs])
        self.t = nn.Sequential(
            nn.Conv1d(ch, 64, 5, 2, 2), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 96, 5, 2, 2), nn.BatchNorm1d(96), nn.GELU(),
            nn.Conv1d(96, 96, 3, 2, 1), nn.BatchNorm1d(96), nn.GELU())
        self.fc = nn.Sequential(nn.Linear(len(specs) * 32 * 2 + 96 * 3, 256), nn.LayerNorm(256),
                                nn.GELU(), nn.Dropout(p),
                                nn.Linear(256, emb), nn.LayerNorm(emb), nn.GELU())
        self.to_lat = nn.Linear(emb, 2 * d)
        self.aux = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, n_target))
        self.d, self.ch = d, ch

    def _embed(self, x, m):
        fbp = []
        for b, bn in zip(self.fb, self.fbn):
            mean, std = masked_mean_std(bn(b(x)).abs(), m)     # per-band energy moments
            fbp += [mean, std]
        tt = self.t(x)
        mt = m[:, :, ::8][:, :, :tt.shape[-1]]
        if mt.shape[-1] < tt.shape[-1]:
            mt = F.pad(mt, (0, tt.shape[-1] - mt.shape[-1]))
        return self.fc(torch.cat(fbp + [masked_stats(tt, mt)], 1))

    def encode(self, x, m):
        """x: (B,C,T) standardized series, m: (B,1,T) validity mask -> (mu, logvar), each (B,d)."""
        mu, logvar = self.to_lat(self._embed(x, m)).chunk(2, dim=1)
        return mu, logvar

    def forward(self, x, m):
        mu, logvar = self.encode(x, m)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu, logvar, self.aux(z)


# --------------------------------------------------------------------------- #
# Loss helpers                                                                #
# --------------------------------------------------------------------------- #
def beta_at(ep, beta_max, warmup):
    """Linear KL warm-up to ``beta_max`` over ``warmup`` epochs."""
    return beta_max * min(1.0, (ep + 1) / max(warmup, 1))


def kl_freebits(mu, logvar, free_bits):
    """KL to N(0, I), with each latent dimension's batch-mean KL clamped from below at ``free_bits``."""
    return torch.clamp(0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).mean(0), min=free_bits).sum()

