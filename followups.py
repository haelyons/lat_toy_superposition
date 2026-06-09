"""Follow-up experiments from the literature critique (CRITIQUE.md), SPEC 9 forks.

Two focused experiments on representative cells spanning the capacity axis
(low-capacity n/m=2 vs high-capacity n/m=8, fixed S=0.9):

  1. epsilon sweep  - is LAT's effect dose-dependent? Larger eps should widen the
     robust basin more, and (in the regime where it concentrates) lower
     interference / raise D more. Tests causality and where the effect saturates.

  2. targeted LAT   - adversary corrupts ONE concept (feature 0). Does targeting
     reproduce the Abbas et al. single-concept concentration, and does it differ
     from untargeted LAT (SPEC 9 predicts "more diffuse")? We track the TARGET
     feature's own geometry, not just the global average.

Writes results/followups.csv (one row per run) and results/followups_summary.csv.
Reuses run_one; only eps_rel / condition vary per run.
"""
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import torch

from config import CFG
from run import run_one


def _agg_concepts(concept_rows, target_feature):
    """Mean r_set2 / pr over represented concepts, plus the target feature's own D."""
    rep = [c for c in concept_rows if c["represented"] == 1]
    import math
    def nanmean(xs):
        xs = [x for x in xs if not math.isnan(x)]
        return sum(xs) / len(xs) if xs else float("nan")
    tgt = next((c for c in concept_rows if c["feature"] == target_feature), None)
    return {
        "r_set2_mean": nanmean([c["r_set2"] for c in rep]),
        "pr_mean": nanmean([c["pr"] for c in rep]),
        "D_target": tgt["D_groundtruth"] if tgt else float("nan"),
        "r_set2_target": tgt["r_set2"] if tgt else float("nan"),
    }


def _worker(job):
    torch.set_num_threads(1)
    n, m, S, cond, seed, eps = job
    cfg = replace(CFG, eps_rel=eps)
    run_row, concept_rows = run_one(n, m, S, cond, seed, cfg)
    agg = _agg_concepts(concept_rows, cfg.target_feature)
    out = {"experiment": job_label(cond, eps),
           "n_over_m": n // m, "sparsity": S, "condition": cond, "eps_rel": eps,
           "seed": seed, "interference": run_row["interference"],
           "mean_D": run_row["mean_D"], "recon_fvu": run_row["recon_fvu"],
           "sae_mono": run_row["sae_mono"], **agg}
    return out


def job_label(cond, eps):
    if cond == "lat":
        return f"eps_sweep(lat,eps={eps:.2f})"
    return cond


def build_jobs():
    """baseline + lat@each-eps + input_at@0.10 + lat_targeted@0.10, per sweep cell/seed."""
    jobs = []
    for (nm, S) in CFG.sweep_cells:
        n, m = nm * CFG.m, CFG.m
        for seed in CFG.sweep_seeds:
            jobs.append((n, m, S, "baseline", seed, 0.10))
            for eps in CFG.sweep_eps:
                jobs.append((n, m, S, "lat", seed, eps))
            jobs.append((n, m, S, "input_at", seed, 0.10))
            jobs.append((n, m, S, "lat_targeted", seed, 0.10))
    return jobs


def main():
    jobs = build_jobs()
    print(f"Follow-up runs: {len(jobs)} "
          f"(cells={CFG.sweep_cells}, eps={CFG.sweep_eps}, seeds={CFG.sweep_seeds})")
    rows = []
    workers = min(4, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_worker, jobs), 1):
            rows.append(r)
            print(f"[{i:2d}/{len(jobs)}] n/m={r['n_over_m']} {r['condition']:12s} "
                  f"eps={r['eps_rel']:.2f} seed={r['seed']} "
                  f"interf={r['interference']:.3f} D={r['mean_D']:.3f} "
                  f"r2={r['r_set2_mean']:.3f}", flush=True)

    fields = list(rows[0].keys())
    with open("results/followups.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("Wrote results/followups.csv")


if __name__ == "__main__":
    main()
