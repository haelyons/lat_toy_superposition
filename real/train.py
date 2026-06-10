"""Training for the three matched regimes on the tiny transformer.

baseline : cross-entropy on the answer token.
input_at : PGD perturbation on the INPUT embeddings (eps relative to ||emb||), predict
           the clean answer  -> the transformer analogue of the toy's input-AT.
lat      : PGD perturbation on the residual stream after layer `resid_layer` (a latent),
           predict the clean answer  -> the transformer analogue of the toy's LAT.

Only the perturbation differs; architecture, data, seed, steps and optimiser are matched.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from real.model import TinyTransformer
from real.task import sample


@dataclass
class TXConfig:
    p: int = 13
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_mlp: int = 256
    # Faithful to practice: a shared CLEAN pretrain (the capability is learned first),
    # then a condition-specific fine-tune that adds the robustness objective. All three
    # conditions branch from the SAME clean checkpoint, so only the objective differs.
    pretrain_steps: int = 1000
    finetune_steps: int = 800
    batch: int = 512
    lr: float = 1e-3
    finetune_lr: float = 3e-4
    eps_rel: float = 0.05            # from-scratch eps=0.1 prevents convergence; AT here
                                     # is a fine-tune, so a smaller ball preserves the skill
    pgd_steps: int = 5
    pgd_step_frac: float = 0.4
    resid_layer: int = 1
    seeds: tuple = (0, 1, 2)
    conditions: tuple = ("baseline", "input_at", "lat")


def _ce(logits, y):
    return F.cross_entropy(logits[:, -1, :], y)


def _pgd_perturb(model, tokens, y, cfg, site):
    """Maximise CE over an L2 ball around the chosen perturbation site; return the delta.
    site = 'emb' (input embeddings) or 'resid' (residual stream at cfg.resid_layer)."""
    if site == "emb":
        base = model.embed(tokens).detach()
        fwd = lambda d: model(tokens, emb_delta=d)
    else:
        base = model.resid_at(tokens, cfg.resid_layer).detach()
        fwd = lambda d: model(tokens, resid_delta=d, resid_layer=cfg.resid_layer)
    flat = base.reshape(base.shape[0], -1)
    eps = cfg.eps_rel * flat.norm(dim=1)                      # (B,)
    step = cfg.pgd_step_frac * eps
    d = torch.zeros_like(base)
    for _ in range(cfg.pgd_steps):
        d = d.detach().requires_grad_(True)
        loss = _ce(fwd(d), y)
        (g,) = torch.autograd.grad(loss, d)
        gf = g.reshape(g.shape[0], -1)
        gn = gf.norm(dim=1, keepdim=True).clamp_min(1e-12)
        df = d.reshape(d.shape[0], -1) + step[:, None] * gf / gn
        # project rows into per-sample L2 balls
        dn = df.norm(dim=1, keepdim=True).clamp_min(1e-12)
        df = df * (eps[:, None] / dn).clamp_max(1.0)
        d = df.reshape(base.shape)
    return d.detach()


def _step_loss(model, x, y, condition, cfg):
    if condition == "baseline":
        return _ce(model(x), y)
    if condition == "input_at":
        d = _pgd_perturb(model, x, y, cfg, "emb")
        return _ce(model(x, emb_delta=d), y)
    if condition == "lat":
        d = _pgd_perturb(model, x, y, cfg, "resid")
        return _ce(model(x, resid_delta=d, resid_layer=cfg.resid_layer), y)
    raise ValueError(condition)


def pretrain_clean(task, split, seed, cfg, steps=None, verbose=False):
    """Shared clean pretrain: learn the capability before any robustness objective."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(1234 + seed)
    model = TinyTransformer(task.vocab, task.seq_len, cfg.d_model, cfg.n_layers,
                            cfg.n_heads, cfg.d_mlp, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for step in range(1, (steps or cfg.pretrain_steps) + 1):
        x, y, _ = sample(split, cfg.batch, gen)
        opt.zero_grad(); loss = _ce(model(x), y); loss.backward(); opt.step()
        if verbose and step % 400 == 0:
            print(f"  [pretrain seed{seed}] step {step} loss {loss.item():.3f}", flush=True)
    return model


def finetune(model, task, split, condition, seed, cfg, steps=None, verbose=False):
    """Condition-specific fine-tune from a clean checkpoint (model is modified in place)."""
    gen = torch.Generator().manual_seed(5678 + seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.finetune_lr)
    for step in range(1, (steps or cfg.finetune_steps) + 1):
        x, y, _ = sample(split, cfg.batch, gen)
        opt.zero_grad()
        loss = _step_loss(model, x, y, condition, cfg)
        loss.backward(); opt.step()
        if verbose and step % 400 == 0:
            print(f"  [{condition} seed{seed}] ft step {step} loss {loss.item():.3f}", flush=True)
    return model


@torch.no_grad()
def accuracy(model, split, task, per_op=False):
    X, Y, OI = split
    pred = model(X)[:, -1, :].argmax(-1)
    correct = (pred == Y)
    if not per_op:
        return correct.float().mean().item()
    out = {}
    for oi in range(task.K):
        m = OI == oi
        out[oi] = correct[m].float().mean().item() if m.any() else float("nan")
    return out
