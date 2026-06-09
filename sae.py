"""TopK SAE on the bottleneck, fixed protocol across conditions (SPEC 6).

Reports FVU (not raw MSE), L0, and 1:1 monosemanticity against the known true
feature directions (columns of W)."""
import torch
import torch.nn as nn


class TopKSAE(nn.Module):
    def __init__(self, m, d_sae, k, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.k = k
        self.b_pre = nn.Parameter(torch.zeros(m))
        self.enc = nn.Linear(m, d_sae)
        self.dec = nn.Linear(d_sae, m, bias=False)
        with torch.no_grad():
            self.enc.weight.copy_(torch.randn(d_sae, m, generator=g) / (m ** 0.5))
            self.dec.weight.copy_(self.enc.weight.t())
            self._normalise_dict()

    def _normalise_dict(self):
        self.dec.weight.div_(self.dec.weight.norm(dim=0, keepdim=True).clamp_min(1e-12))

    def encode(self, h):
        z = torch.relu(self.enc(h - self.b_pre))
        topv, topi = z.topk(self.k, dim=1)
        z_sparse = torch.zeros_like(z).scatter_(1, topi, topv)
        return z_sparse

    def forward(self, h):
        z = self.encode(h)
        return self.dec(z) + self.b_pre, z


def train_sae(h_data, m, d_sae, k, cfg, seed=0):
    sae = TopKSAE(m, d_sae, k, seed=seed)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.sae_lr)
    g = torch.Generator().manual_seed(20_000 + seed)
    N = h_data.shape[0]
    for _ in range(cfg.sae_steps):
        idx = torch.randint(0, N, (cfg.sae_batch,), generator=g)
        h = h_data[idx]
        recon, _ = sae(h)
        loss = ((recon - h) ** 2).sum(dim=1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            sae._normalise_dict()
    return sae


@torch.no_grad()
def fvu(sae, h):
    recon, _ = sae(h)
    mse = ((recon - h) ** 2).sum(dim=1).mean()
    var = ((h - h.mean(dim=0)) ** 2).sum(dim=1).mean()
    return (mse / var.clamp_min(1e-12)).item()


@torch.no_grad()
def l0(sae, h, thresh=1e-6):
    _, z = sae(h)
    return (z > thresh).float().sum(dim=1).mean().item()


@torch.no_grad()
def monosemanticity(sae, W):
    """Mean max-cosine between true feature directions (cols of W, m-dim) and
    SAE dictionary atoms (cols of dec.weight, m-dim). 1.0 => perfect 1:1 recovery."""
    feats = W / W.norm(dim=0, keepdim=True).clamp_min(1e-12)      # (m, n)
    atoms = sae.dec.weight / sae.dec.weight.norm(dim=0, keepdim=True).clamp_min(1e-12)  # (m, d)
    cos = feats.t() @ atoms                                       # (n, d)
    return cos.abs().max(dim=1).values.mean().item()
