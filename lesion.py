"""Lesion / editability probe -- does "more concentrated" mean "more surgically
editable"? (follow-up to Claim B + the innovability probe.)

Claim B established that LAT *concentrates* the weight geometry: per-concept
dimensionality D up, off-diagonal interference I down. But concentration was only ever
measured as GEOMETRY. Its assumed *meaning* -- that a low-interference concept can be
modified without collateral damage to the others -- was never tested. This closes that
IOU: it performs a real capability modification (directional ablation = concept knockout,
the standard activation-edit) and measures the collateral damage to the OTHER concepts.

For concept i with unit write-direction w_i in the early latent, ablation removes its
component:  h' = h - (h . w_i) w_i ; x' = decode(h'). The damage to concept j is
mediated by (w_i . w_j) -- exactly the term summed in concept i's interference I_i. So
there is an analytic expectation (collateral_i tracks I_i); the experiment asks whether
it cashes out in FUNCTION (reconstruction FVU) and whether LAT's lower I yields lower
collateral -- i.e. whether LAT models are more surgically editable.

This is the "edit / remove an EXISTING capability" cell of the 2x2 the innovability
probe opened ("acquire a NEW capability"). Standard SPEC training (geometric importance)
so the concentration values match Claim B. Writes results/lesion.csv.
"""
import csv
import os
from concurrent.futures import ProcessPoolExecutor

import torch

from config import CFG
from train import train
from data import sample_batch


@torch.no_grad()
def per_concept_interference(W):
    """I_i = mean_{j!=i} (w_hat_i . w_hat_j)^2  -- row i of the normalised-Gram
    off-diagonal (the per-concept version of metrics.offdiag_interference)."""
    Wn = W / W.norm(dim=0, keepdim=True).clamp_min(1e-12)
    G = (Wn.t() @ Wn) ** 2
    G.fill_diagonal_(0.0)
    return G.sum(dim=1) / (G.shape[0] - 1)            # (n,)


@torch.no_grad()
def per_concept_dim(W):
    """D_i = ||W_i||^2 / sum_j (W_hat_i . W_j)^2 (metrics.ground_truth_dimensionality)."""
    norms = W.norm(dim=0)
    What = W / norms.clamp_min(1e-12)
    proj = What.t() @ W
    denom = (proj ** 2).sum(dim=1).clamp_min(1e-12)
    return norms ** 2 / denom


@torch.no_grad()
def _fvu_per_col(x, xh):
    """Per-column FVU (n,): unexplained variance of each concept's reconstruction."""
    num = ((x - xh) ** 2).mean(0)
    den = ((x - x.mean(0)) ** 2).mean(0).clamp_min(1e-12)
    return num / den


@torch.no_grad()
def _ablate(model, x, i):
    """Directional knockout of concept i in the early latent; return reconstruction."""
    h = model.encode(x)
    wi = model.W[:, i]
    wihat = wi / wi.norm().clamp_min(1e-12)
    coeff = h @ wihat                                 # (B,)
    h2 = h - coeff[:, None] * wihat[None, :]
    return model.decode(h2)


def run_cell(n, m, sparsity, condition, seed, cfg):
    model, importance = train(n, m, sparsity, condition, seed, cfg)
    W = model.W.detach()
    wnorm = W.norm(dim=0)
    I = per_concept_interference(W)
    D = per_concept_dim(W)
    represented = wnorm >= cfg.represented_w_norm     # only concepts the model encodes

    egen = torch.Generator().manual_seed(80_000 + seed)
    x = sample_batch(cfg.eval_batch, n, sparsity, egen)
    clean = _fvu_per_col(x, model(x))                 # (n,)

    rep_idx = torch.nonzero(represented, as_tuple=True)[0].tolist()
    rows = []
    for i in rep_idx:
        ab = _fvu_per_col(x, _ablate(model, x, i))    # (n,)
        delta = ab - clean                            # FVU increase from the knockout
        # collateral = damage to the OTHER represented concepts
        mask = represented.clone(); mask[i] = False
        collateral = float(delta[mask].mean().item()) if mask.any() else float("nan")
        rows.append({
            "n_over_m": n // m, "sparsity": sparsity, "condition": condition,
            "seed": seed, "feature": i,
            "w_norm": float(wnorm[i].item()),
            "I_i": float(I[i].item()), "D_i": float(D[i].item()),
            "self_drop": float(delta[i].item()),       # capability actually removed (large +)
            "collateral": collateral,                  # side-effect on the rest
            "n_represented": len(rep_idx),
        })
    return rows


def _worker(job):
    torch.set_num_threads(1)
    n, m, S, cond, seed = job
    return run_cell(n, m, S, cond, seed, CFG)


def main():
    cells = [(nm, S) for nm in CFG.n_over_m for S in CFG.sparsity]   # full Claim B grid
    seeds = CFG.seeds
    jobs = [(nm * CFG.m, CFG.m, S, cond, seed)
            for (nm, S) in cells for cond in CFG.conditions for seed in seeds]
    print(f"Lesion runs: {len(jobs)} (cells={cells}, conds={CFG.conditions}, seeds={seeds})")
    all_rows = []
    workers = min(4, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, rows in enumerate(ex.map(_worker, jobs), 1):
            all_rows.extend(rows)
            if rows:
                r0 = rows[0]
                col = sum(r["collateral"] for r in rows) / len(rows)
                sd = sum(r["self_drop"] for r in rows) / len(rows)
                print(f"[{i:3d}/{len(jobs)}] n/m={r0['n_over_m']} S={r0['sparsity']} "
                      f"{r0['condition']:9s} seed={r0['seed']} "
                      f"self_drop={sd:.2f} collateral={col:.4f} (rep={r0['n_represented']})",
                      flush=True)
    with open("results/lesion.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    print(f"Wrote results/lesion.csv ({len(all_rows)} concept-lesions)")


if __name__ == "__main__":
    main()
