"""Selectivity ablation — is untargeted LAT's concentration effect UNIFORM across
concepts, or GRADED by a concept's importance / frequency / reliance?

Motivation: Abbas et al. apply *general* (untargeted) LAT and find one concept (refusal)
concentrates. If LAT concentrated everything uniformly, that would be unremarkable; the
interesting (and publishable) claim is that general LAT is general in its PERTURBATION but
SELECTIVE in its EFFECT — it concentrates the concepts the loss most depends on. This
probe tests that directly: it regresses the LAT-induced change in concentration, per
concept, against the concept's importance (geometric decay, varies within a cell) and
frequency (1 - sparsity, varies across cells).

Concentration is read TWO ways:
  - weight geometry (non-degenerate): per-concept dimensionality D_i (higher = cleaner),
    per-concept interference I_i (lower = cleaner), and w_norm (reliance/magnitude).
  - LATE-site activation effective dimensionality (the Abbas-style SVD measure, but at the
    post-nonlinear h2 of the two-layer toy, where the cloud is NOT structurally rank-1 —
    unlike the disqualified early-h1 proxy): top1_evr (higher = more 1D), pr (lower = more
    concentrated).

Writes results/selectivity.csv (one row per represented concept per model).
"""
import csv
import os
from concurrent.futures import ProcessPoolExecutor

import torch

from config import CFG
from train import train
from data import sample_batch, geometric_importance
from metrics import ground_truth_dimensionality
from lesion import per_concept_interference


@torch.no_grad()
def _late_h2(model, x):
    """Post-nonlinear late latent of the two-layer toy: h2 = ReLU(W2 h1 + c)."""
    h1 = model.encode(x)
    return torch.relu(h1 @ model.W2.t() + model.c)


@torch.no_grad()
def _cloud_concentration(H, x):
    """Per-concept effective-dim of the activation-difference cloud at site H.
    top1_evr (higher=more 1D/concentrated), pr (lower=more concentrated)."""
    n = x.shape[1]
    active = x > 0
    t1 = torch.full((n,), float("nan"))
    pr = torch.full((n,), float("nan"))
    for i in range(n):
        m = active[:, i]
        if m.sum() < 8 or (~m).sum() < 8:
            continue
        cloud = H[m] - H[~m].mean(0)
        lam = torch.linalg.svdvals(cloud) ** 2
        tot = lam.sum().clamp_min(1e-12)
        t1[i] = (lam[0] / tot)
        pr[i] = (lam.sum() ** 2) / (lam ** 2).sum().clamp_min(1e-12)
    return t1, pr


def run_cell(n, m, sparsity, condition, seed, cfg):
    model, importance = train(n, m, sparsity, condition, seed, cfg)
    W = model.W.detach()
    wnorm = W.norm(dim=0)
    D = ground_truth_dimensionality(W)
    I = per_concept_interference(W)
    imp = geometric_importance(n, cfg.importance_decay)

    egen = torch.Generator().manual_seed(90_000 + seed)
    x = sample_batch(cfg.eval_batch, n, sparsity, egen)
    t1_late, pr_late = _cloud_concentration(_late_h2(model, x), x)

    rows = []
    for i in range(n):
        if wnorm[i] < cfg.represented_w_norm:
            continue
        rows.append({
            "n_over_m": n // m, "sparsity": sparsity, "frequency": round(1 - sparsity, 4),
            "condition": condition, "seed": seed, "feature": i,
            "importance": float(imp[i].item()), "imp_rank": i,        # low rank = important
            "w_norm": float(wnorm[i].item()), "D_i": float(D[i].item()),
            "I_i": float(I[i].item()),
            "late_top1evr": float(t1_late[i].item()), "late_pr": float(pr_late[i].item()),
        })
    return rows


def _worker(job):
    torch.set_num_threads(1)
    n, m, S, cond, seed = job
    return run_cell(n, m, S, cond, seed, CFG)


def main():
    cells = [(nm, S) for nm in (2, 4, 8) for S in (0.8, 0.9, 0.99)]   # importance × frequency
    seeds = (0, 1, 2)
    conds = ("baseline", "input_at", "lat")
    jobs = [(nm * CFG.m, CFG.m, S, c, s) for (nm, S) in cells for c in conds for s in seeds]
    print(f"Selectivity runs: {len(jobs)} (cells={cells}, conds={conds}, seeds={seeds})")
    all_rows = []
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as ex:
        for i, rows in enumerate(ex.map(_worker, jobs), 1):
            all_rows.extend(rows)
            if rows:
                r0 = rows[0]
                print(f"[{i:3d}/{len(jobs)}] n/m={r0['n_over_m']} S={r0['sparsity']} "
                      f"{r0['condition']:9s} seed={r0['seed']} ({len(rows)} represented)",
                      flush=True)
    with open("results/selectivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    print(f"Wrote results/selectivity.csv ({len(all_rows)} concept-rows)")


if __name__ == "__main__":
    main()
