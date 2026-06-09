"""Training loops for the three matched conditions (SPEC 5).

baseline  : standard reconstruction.
input_at  : PGD perturbation on x (L2 ball, eps relative to ||x||), reconstruct clean x.
lat       : PGD perturbation on the bottleneck h (L2, eps relative to ||h||),
            min_theta max_{||d||<=eps} loss(x, decode(h + d)).

Only the perturbation differs; architecture, data stream, seed, step count and
optimiser are identical across conditions.
"""
import torch

from data import (ToyModel, TwoLayerToyModel, geometric_importance,
                  sample_batch, weighted_mse)


def build_model(n, m, seed, cfg):
    return (TwoLayerToyModel if cfg.two_layer else ToyModel)(n, m, seed=seed)


def _l2_project(delta, eps_per_sample):
    """Project rows of delta into per-sample L2 balls of radius eps_per_sample (B,1)."""
    norm = delta.norm(dim=1, keepdim=True).clamp_min(1e-12)
    factor = (eps_per_sample / norm).clamp_max(1.0)
    return delta * factor


def _pgd(loss_fn, delta0, eps_per_sample, step_size, steps):
    """Maximise loss_fn(delta) over the L2 ball via projected gradient ascent."""
    delta = delta0.clone()
    for _ in range(steps):
        delta = delta.detach().requires_grad_(True)
        loss = loss_fn(delta)
        (grad,) = torch.autograd.grad(loss, delta)
        gnorm = grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
        delta = delta + step_size * grad / gnorm
        delta = _l2_project(delta, eps_per_sample)
    return delta.detach()


def train(n, m, sparsity, condition, seed, cfg):
    """Train one (cell, condition, seed). Returns the trained model and the importance vector."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(10_000 + seed)
    model = build_model(n, m, seed, cfg)
    importance = geometric_importance(n, cfg.importance_decay)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    step_frac = cfg.pgd_step_frac

    for _ in range(cfg.steps):
        x = sample_batch(cfg.batch, n, sparsity, gen)
        opt.zero_grad()

        if condition == "baseline":
            loss = weighted_mse(x, model(x), importance)

        elif condition == "input_at":
            eps = cfg.eps_rel * x.norm(dim=1, keepdim=True)
            step_size = step_frac * eps
            delta0 = torch.zeros_like(x)

            def loss_fn(d):
                return weighted_mse(x, model.decode(model.encode(x + d)), importance)

            delta = _pgd(loss_fn, delta0, eps, step_size, cfg.pgd_steps)
            loss = weighted_mse(x, model.decode(model.encode(x + delta)), importance)

        elif condition == "lat":
            with torch.no_grad():
                h = model.encode(x)
            eps = cfg.eps_rel * h.norm(dim=1, keepdim=True)
            step_size = step_frac * eps
            delta0 = torch.zeros_like(h)

            def loss_fn(d):
                return weighted_mse(x, model.decode(h + d), importance)

            delta = _pgd(loss_fn, delta0, eps, step_size, cfg.pgd_steps)
            # outer step: recompute h WITH grad so theta sees the perturbation
            loss = weighted_mse(x, model.decode(model.encode(x) + delta), importance)
        else:
            raise ValueError(condition)

        loss.backward()
        opt.step()

    return model, importance
