"""Evolvability probe (SPEC 10, deferred from v1) -- the Wagner 'robustness enables
evolvability' half of the hypothesis.

Robustness (a wide, clean neutral basin) is established. Wagner's claim is that the
same neutral region makes reaching a NEW phenotype easier. We test two facets:

  Part A -- COMPOSITIONALITY (cheap): evaluate trained models on novel feature
    COMBINATIONS (more features co-active than seen in training). Better graceful
    degradation = better access to neighbouring phenotypes via composition.

  Part B -- ADAPT-TO-NEW-CONCEPT (the Wagner-faithful test): pretrain with one
    feature held inactive, then fine-tune it in IDENTICALLY across conditions
    (only the pretrained init differs). Measure adaptation speed, final quality,
    and forgetting of old concepts.

Prediction: LAT (wider/cleaner basin, less interference -> more orthogonal room)
adapts faster with less forgetting, MOST at mid capacity, vanishing at n/m=8
(the Bereska saturation boundary). Falsifiable: a flat basin could instead be
STIFF (slower adaptation) -- which would decouple robustness from evolvability.

Uniform importance throughout (SPEC 9 fork) so the 'new' concept is on equal
footing with the others. Writes results/evolvability_compositionality.csv and
results/evolvability_adaptation.csv.
"""
import csv
import os
from concurrent.futures import ProcessPoolExecutor

import torch

from config import CFG
from train import build_model, _pgd
from data import sample_batch, weighted_mse


def _sample(batch, n, sparsity, gen, holdout=None):
    x = sample_batch(batch, n, sparsity, gen)
    if holdout is not None:
        x[:, holdout] = 0.0
    return x


def _sample_adapt(batch, n, sparsity, gen, newfeat, p_new):
    """Phase-2 sampler: old features at training sparsity, the NEW feature active
    with elevated prob p_new (selection pressure -- the new function now matters)."""
    x = sample_batch(batch, n, sparsity, gen)
    on = (torch.rand(batch, generator=gen) < p_new).float()
    x[:, newfeat] = on * torch.rand(batch, generator=gen)
    return x


def _sample_kactive(batch, n, k, gen):
    """Exactly k features active per row (novel dense combinations), value U(0,1)."""
    x = torch.zeros(batch, n)
    for r in range(batch):
        idx = torch.randperm(n, generator=gen)[:k]
        x[r, idx] = torch.rand(k, generator=gen)
    return x


@torch.no_grad()
def _fvu_cols(model, x, cols):
    xh = model(x)
    num = ((x[:, cols] - xh[:, cols]) ** 2).mean()
    den = ((x[:, cols] - x[:, cols].mean(0)) ** 2).mean().clamp_min(1e-12)
    return (num / den).item()


def _adv_delta(model, x, importance, cfg, condition):
    """One adversarial perturbation for the matched condition (used in pretrain)."""
    if condition == "input_at":
        eps = cfg.eps_rel * x.norm(dim=1, keepdim=True)
        step = cfg.pgd_step_frac * eps
        loss_fn = lambda d: weighted_mse(x, model.decode(model.encode(x + d)), importance)
        d = _pgd(loss_fn, torch.zeros_like(x), eps, step, cfg.pgd_steps)
        return ("input", d)
    if condition == "lat":
        with torch.no_grad():
            h = model.encode(x)
        eps = cfg.eps_rel * h.norm(dim=1, keepdim=True)
        step = cfg.pgd_step_frac * eps
        loss_fn = lambda d: weighted_mse(x, model.decode(h + d), importance)
        d = _pgd(loss_fn, torch.zeros_like(h), eps, step, cfg.pgd_steps)
        return ("latent", d)
    return (None, None)


def _train_steps(model, opt, n, sparsity, importance, cfg, steps, gen,
                 condition="baseline", holdout=None):
    """Matched training loop (mirrors train.py) with optional held-out feature."""
    for _ in range(steps):
        x = _sample(cfg.batch, n, sparsity, gen, holdout=holdout)
        opt.zero_grad()
        kind, d = _adv_delta(model, x, importance, cfg, condition)
        if kind == "input":
            loss = weighted_mse(x, model.decode(model.encode(x + d)), importance)
        elif kind == "latent":
            loss = weighted_mse(x, model.decode(model.encode(x) + d), importance)
        else:
            loss = weighted_mse(x, model(x), importance)
        loss.backward()
        opt.step()


def run_cell(n, m, sparsity, condition, seed, cfg, adapt_steps=1500, log_every=150):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(10_000 + seed)
    importance = torch.ones(n)                      # uniform (SPEC 9 fork)
    holdout = n - 1                                  # the 'new' concept
    old_cols = list(range(n - 1))

    # ---- Phase 1: pretrain with feature `holdout` held inactive ----
    model = build_model(n, m, seed, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    _train_steps(model, opt, n, sparsity, importance, cfg, cfg.steps, gen,
                 condition=condition, holdout=holdout)

    # eval batches
    egen = torch.Generator().manual_seed(40_000 + seed)
    x_full = _sample(cfg.eval_batch, n, sparsity, egen)          # all features active-able
    x_old = _sample(cfg.eval_batch, n, sparsity, egen, holdout=holdout)

    fvu_new_start = _fvu_cols(model, x_full, [holdout])
    fvu_old_start = _fvu_cols(model, x_old, old_cols)
    wnorm_new_start = float(model.W[:, holdout].norm().item())

    # ---- Part A: compositionality (novel dense combinations), pre-adaptation ----
    comp_rows = []
    cgen = torch.Generator().manual_seed(50_000 + seed)
    for k in [1, 2, 4, 8, min(16, n)]:
        if k > n - 1:
            continue
        xk = _sample_kactive(2048, n - 1, k, cgen)              # only old features
        xk_full = torch.zeros(2048, n); xk_full[:, :n - 1] = xk
        comp_rows.append({"n_over_m": n // m, "sparsity": sparsity, "condition": condition,
                          "seed": seed, "k_active": k,
                          "fvu_old": _fvu_cols(model, xk_full, old_cols)})

    # Give the new concept a FRESH, identical readout (column + bias) across all
    # conditions, so its own parameters don't start dead (held-out pretraining drives
    # the ReLU readout off -> zero gradient -> un-revivable, a coin-flip confound).
    # After this reset the ONLY difference between conditions is the surrounding basin
    # geometry the new concept must integrate into -- which is what evolvability tests.
    with torch.no_grad():
        rg = torch.Generator().manual_seed(70_000 + seed)
        model.W.data[:, holdout] = torch.randn(m, generator=rg) / (m ** 0.5)
        model.b.data[holdout] = 0.0

    # ---- Phase 2: adapt -- select for the new concept, identical SGD across conditions.
    # New feature now frequent (p_new) and important (imp_new) = selection pressure;
    # only the pretrained init differs between conditions.
    p_new, imp_new = 0.5, 3.0
    imp_adapt = torch.ones(n); imp_adapt[holdout] = imp_new
    agen = torch.Generator().manual_seed(60_000 + seed)
    opt2 = torch.optim.Adam(model.parameters(), lr=cfg.lr)       # fresh optimiser
    x_adapt_eval = _sample_adapt(cfg.eval_batch, n, sparsity, egen, holdout, p_new)
    traj = []
    steps_to_thresh = None
    for step in range(0, adapt_steps + 1, log_every):
        if step > 0:
            for _ in range(log_every):
                xb = _sample_adapt(cfg.batch, n, sparsity, agen, holdout, p_new)
                opt2.zero_grad()
                loss = weighted_mse(xb, model(xb), imp_adapt)   # standard recon, no adversary
                loss.backward(); opt2.step()
        fnew = _fvu_cols(model, x_adapt_eval, [holdout])
        fold = _fvu_cols(model, x_old, old_cols)
        traj.append((step, fnew, fold))
        if steps_to_thresh is None and fnew < 0.20:
            steps_to_thresh = step

    fvu_new_start = traj[0][1]                  # start measured on the adapt distribution
    fvu_new_end = traj[-1][1]
    fvu_old_end = traj[-1][2]
    adapt_row = {
        "n_over_m": n // m, "sparsity": sparsity, "condition": condition, "seed": seed,
        "fvu_new_start": fvu_new_start, "fvu_new_end": fvu_new_end,
        "fvu_old_start": fvu_old_start, "fvu_old_end": fvu_old_end,
        "forgetting": fvu_old_end - fvu_old_start,             # >0 = old concepts degraded
        "steps_to_new_thresh": steps_to_thresh if steps_to_thresh is not None else -1,
        "wnorm_new_start": wnorm_new_start,
        "wnorm_new_end": float(model.W[:, holdout].norm().item()),
        "traj": ";".join(f"{s}:{fn:.3f}:{fo:.3f}" for s, fn, fo in traj),
    }
    return comp_rows, adapt_row


def _worker(job):
    torch.set_num_threads(1)
    n, m, S, cond, seed = job
    return run_cell(n, m, S, cond, seed, CFG)


def main():
    cells = [(2, 0.9), (4, 0.9), (8, 0.9)]          # span capacity at fixed sparsity
    seeds = (0, 1, 2, 3, 4)                          # >=5 seeds (SPEC 7); evolvability signals are seed-noisy
    jobs = [(nm * CFG.m, CFG.m, S, cond, seed)
            for (nm, S) in cells for cond in CFG.conditions for seed in seeds]
    print(f"Evolvability runs: {len(jobs)} (cells={cells}, conds={CFG.conditions}, seeds={seeds})")
    comp_all, adapt_all = [], []
    workers = min(4, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (comp_rows, adapt_row) in enumerate(ex.map(_worker, jobs), 1):
            comp_all.extend(comp_rows); adapt_all.append(adapt_row)
            a = adapt_row
            print(f"[{i:2d}/{len(jobs)}] n/m={a['n_over_m']} {a['condition']:9s} seed={a['seed']} "
                  f"new_fvu {a['fvu_new_start']:.2f}->{a['fvu_new_end']:.2f} "
                  f"steps2thresh={a['steps_to_new_thresh']} forget={a['forgetting']:+.3f}",
                  flush=True)
    with open("results/evolvability_compositionality.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comp_all[0].keys())); w.writeheader(); w.writerows(comp_all)
    with open("results/evolvability_adaptation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(adapt_all[0].keys())); w.writeheader(); w.writerows(adapt_all)
    print("Wrote results/evolvability_{compositionality,adaptation}.csv")


if __name__ == "__main__":
    main()
