"""Orchestrate the full sweep: grid cell x condition x seed (SPEC 7, 8).

For every run, train the model and log all Section-6 metrics:
  - ground-truth superposition (D_i, off-diagonal interference)
  - concept proxies (diff-of-means -> top-1 EVR, PR)
  - SAE proxies (FVU, L0, monosemanticity) on clean vs adversarial latents
  - robust-region critical radii r_i along trained vs held-out directions

Outputs two tidy CSVs under results/:
  runs.csv      - one row per (cell, condition, seed): model-level scalars
  concepts.csv  - one row per (cell, condition, seed, feature): per-concept arrays
"""
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import torch

from config import CFG
from data import sample_batch, weighted_mse
from train import train, _pgd, _l2_project  # noqa: F401 (reuse PGD for eval attack)
from metrics import (ground_truth_dimensionality, offdiag_interference,
                     concept_proxies, robust_region)
from sae import train_sae, fvu, l0, monosemanticity


def adversarial_latents(model, x, importance, cfg):
    """Frozen-model latent PGD attack (LAT-style) to get adversarial bottleneck h."""
    with torch.no_grad():
        h = model.encode(x)
    eps = cfg.eps_rel * h.norm(dim=1, keepdim=True)
    step_size = cfg.pgd_step_frac * eps

    def loss_fn(d):
        return weighted_mse(x, model.decode(h + d), importance)

    delta = _pgd(loss_fn, torch.zeros_like(h), eps, step_size, cfg.pgd_steps)
    return (h + delta).detach(), h.detach()


@torch.no_grad()
def eval_fvu(model, x, importance):
    x_hat = model(x)
    mse = weighted_mse(x, x_hat, importance).item()
    var = (((x - x.mean(0)) ** 2) * importance).sum(1).mean().item()
    return mse / max(var, 1e-12)


def run_one(n, m, sparsity, condition, seed, cfg):
    model, importance = train(n, m, sparsity, condition, seed, cfg)

    gen = torch.Generator().manual_seed(40_000 + seed)
    x_eval = sample_batch(cfg.eval_batch, n, sparsity, gen)

    W = model.W.detach()
    D = ground_truth_dimensionality(W)
    interference = offdiag_interference(W)
    proxies = concept_proxies(model, x_eval)
    robust = robust_region(model, x_eval, cfg, seed=seed)

    # SAE on clean latents; evaluate on clean vs adversarial latents
    with torch.no_grad():
        h_clean = model.encode(x_eval)
    k = max(1, round(n * (1.0 - sparsity)))
    d_sae = cfg.sae_dict_mult * n
    sae = train_sae(h_clean, m, d_sae, k, cfg, seed=seed)
    h_adv, _ = adversarial_latents(model, x_eval, importance, cfg)

    run_row = {
        "n": n, "m": m, "n_over_m": n // m, "sparsity": sparsity,
        "condition": condition, "seed": seed,
        "recon_fvu": eval_fvu(model, x_eval, importance),
        "interference": interference,
        "sae_fvu_clean": fvu(sae, h_clean),
        "sae_fvu_adv": fvu(sae, h_adv),
        "sae_l0_clean": l0(sae, h_clean),
        "sae_l0_adv": l0(sae, h_adv),
        "sae_mono": monosemanticity(sae, W),
        "sae_k": k, "sae_dict": d_sae,
        "mean_D": float(torch.nanmean(D).item()),
    }

    concept_rows = []
    for i in range(n):
        w_norm = float(W[:, i].norm().item())
        concept_rows.append({
            "n": n, "m": m, "n_over_m": n // m, "sparsity": sparsity,
            "condition": condition, "seed": seed, "feature": i,
            "w_norm": w_norm,
            "represented": int(w_norm >= cfg.represented_w_norm),
            "D_groundtruth": float(D[i].item()),
            "top1_evr": float(proxies["top1_evr"][i].item()),
            "pr": float(proxies["pr"][i].item()),
            "r_set1": float(robust["r_set1"][i].item()),
            "r_set2": float(robust["r_set2"][i].item()),
            "sat_set1": float(robust["sat_set1"][i].item()),
            "sat_set2": float(robust["sat_set2"][i].item()),
        })
    return run_row, concept_rows


def _worker(job):
    # tiny matrices: 1 thread/process is fastest; parallelism comes from many processes
    torch.set_num_threads(1)
    n, m, S, cond, seed = job
    return run_one(n, m, S, cond, seed, CFG)


def main():
    cfg = CFG
    t0 = time.time()
    runs, concepts = [], []
    cells = [(r * cfg.m, cfg.m, S) for r in cfg.n_over_m for S in cfg.sparsity]
    jobs = [(n, m, S, cond, seed)
            for (n, m, S) in cells
            for cond in cfg.conditions
            for seed in cfg.seeds]
    total = len(jobs)
    done = 0
    workers = min(4, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for rr, cr in ex.map(_worker, jobs):
            runs.append(rr)
            concepts.extend(cr)
            done += 1
            el = time.time() - t0
            print(f"[{done:3d}/{total}] n={rr['n']} S={rr['sparsity']} "
                  f"{rr['condition']} seed={rr['seed']} "
                  f"recon_fvu={rr['recon_fvu']:.4f} ({el:.0f}s)", flush=True)

    with open("results/runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0].keys()))
        w.writeheader()
        w.writerows(runs)
    with open("results/concepts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(concepts[0].keys()))
        w.writeheader()
        w.writerows(concepts)
    with open("results/config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print(f"Done {total} runs in {time.time() - t0:.0f}s -> results/runs.csv, results/concepts.csv")


if __name__ == "__main__":
    main()
