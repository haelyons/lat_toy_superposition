"""Innovability probe (SPEC §10, follow-up) -- the resolution of the evolvability
question the single-concept probe left split.

`evolvability.py` found the two faces of Wagner come APART in this toy: LAT's robust
basin forgets less (protects existing concepts) but is STIFFER (slower to acquire one
new concept). Reported as two traded-off signals, that is a half-result.

But Wagner's "evolvability" is neither speed nor a single target. It is INNOVABILITY:
the breadth of NEW phenotypes reachable WITHOUT sacrificing existing function. Evolution
selects against a mutation that grants a new function but breaks an essential one. That
joint criterion -- innovate while remaining viable -- collapses the two faces into one
number, and is the faithful LLM-context reading: can the network acquire new capabilities
without catastrophic forgetting of old ones?

Design: pretrain on a SUB-environment (a set H of features held inactive). Each held-out
feature is a candidate "new niche". From the SAME pretrained checkpoint, adapt each
candidate independently under selection pressure (identical SGD across conditions; only
the pretrained basin differs). Across the battery we measure, as a function of step budget:

  raw innovability   = fraction of the battery reached (new-concept FVU < THRESH).
                       Stiffness should HURT LAT here (slower descent).
  clean innovability  = fraction reached WITH bounded forgetting (old-concept FVU rise < DELTA).
                       LAT's preservation should WIN here.
  grad_norm_init      = gradient magnitude on the new concept at adaptation step 0
                       (direct stiffness measurement, not inferred from step counts).

Prediction that PROVES the analogy: LAT clean-innovability >= baseline even where its
raw/fast innovability is worse -- robustness enables VIABLE innovation. The opposite
(LAT worse on both) is a clean refutation. Either is decisive.

Uniform importance (SPEC §9 fork). Writes results/innovability.csv.
"""
import copy
import csv
import os
from concurrent.futures import ProcessPoolExecutor

import torch

from config import CFG
from train import build_model
from data import sample_batch, weighted_mse
from evolvability import _adv_delta, _fvu_cols

THRESH = 0.20          # new-concept FVU below this = "reached" (matches evolvability.py)
DELTA = 0.05           # old-concept FVU rise below this = "viable" (function preserved)
BATTERY = 6            # number of candidate new concepts (held-out niches)
# Log-spaced checkpoints: dense early to expose the stiffness-driven speed gap (raw
# innovability), sparse late to capture clean convergence. Budget = 1500 SGD steps.
CHECKPOINTS = [0, 25, 50, 75, 100, 150, 250, 400, 700, 1100, 1500]
P_NEW, IMP_NEW = 0.5, 3.0


def _sample_holdout(batch, n, sparsity, gen, holdouts):
    """Old features at training sparsity; every held-out feature forced inactive."""
    x = sample_batch(batch, n, sparsity, gen)
    x[:, holdouts] = 0.0
    return x


def _sample_adapt_one(batch, n, sparsity, gen, holdouts, newfeat):
    """Phase-2 sampler for ONE candidate: old features normal, the candidate `newfeat`
    active w.p. P_NEW (selection pressure), all OTHER held-out niches stay dormant."""
    x = sample_batch(batch, n, sparsity, gen)
    x[:, holdouts] = 0.0
    on = (torch.rand(batch, generator=gen) < P_NEW).float()
    x[:, newfeat] = on * torch.rand(batch, generator=gen)
    return x


def _pretrain(n, m, sparsity, condition, seed, cfg, holdouts):
    """Phase 1: matched training on the sub-environment (holdouts inactive)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(10_000 + seed)
    importance = torch.ones(n)                          # uniform (SPEC §9 fork)
    model = build_model(n, m, seed, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for _ in range(cfg.steps):
        x = _sample_holdout(cfg.batch, n, sparsity, gen, holdouts)
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
    return model, importance


def _grad_norm_new(model, x, imp, newfeat):
    """L2 norm of the loss gradient on the new concept's parameters at the current
    point -- a direct stiffness reading (small = stiff basin, slow to move)."""
    model.zero_grad(set_to_none=True)
    loss = weighted_mse(x, model(x), imp)
    g_col = torch.autograd.grad(loss, model.W, retain_graph=False)[0][:, newfeat]
    return float(g_col.norm().item())


def _adapt_candidate(base_model, n, m, sparsity, seed, newfeat, holdouts, old_cols, egen):
    """From a COPY of the pretrained model, integrate one new niche. Returns trajectory
    of (step, fvu_new, fvu_old) plus the init gradient norm."""
    model = copy.deepcopy(base_model)
    # Fresh, identical readout for the new concept across all conditions, so its own
    # parameters don't start dead (held-out pretraining drives the ReLU readout off ->
    # zero gradient -> un-revivable confound). Only the SURROUNDING basin then differs.
    with torch.no_grad():
        rg = torch.Generator().manual_seed(70_000 + seed * 100 + newfeat)
        model.W.data[:, newfeat] = torch.randn(m, generator=rg) / (m ** 0.5)
        model.b.data[newfeat] = 0.0

    imp = torch.ones(n); imp[newfeat] = IMP_NEW
    agen = torch.Generator().manual_seed(60_000 + seed * 100 + newfeat)
    opt = torch.optim.Adam(model.parameters(), lr=CFG.lr)

    # held-out eval distributions (old concepts; the candidate under selection)
    x_old = _sample_holdout(CFG.eval_batch, n, sparsity, egen, holdouts)
    x_new_eval = _sample_adapt_one(CFG.eval_batch, n, sparsity, egen, holdouts, newfeat)

    grad_init = _grad_norm_new(model, x_new_eval, imp, newfeat)
    fvu_old_start = _fvu_cols(model, x_old, old_cols)

    traj = []
    prev = 0
    for step in CHECKPOINTS:
        for _ in range(step - prev):
            xb = _sample_adapt_one(CFG.batch, n, sparsity, agen, holdouts, newfeat)
            opt.zero_grad()
            loss = weighted_mse(xb, model(xb), imp)            # standard recon, no adversary
            loss.backward(); opt.step()
        prev = step
        fnew = _fvu_cols(model, x_new_eval, [newfeat])
        fold = _fvu_cols(model, x_old, old_cols)
        traj.append((step, fnew, fold))
    return traj, grad_init, fvu_old_start


def run_cell(n, m, sparsity, condition, seed, cfg):
    holdouts = list(range(n - BATTERY, n))             # last BATTERY features = new niches
    old_cols = list(range(n - BATTERY))
    base, _ = _pretrain(n, m, sparsity, condition, seed, cfg, holdouts)
    egen = torch.Generator().manual_seed(40_000 + seed)

    rows = []
    for newfeat in holdouts:
        traj, grad_init, fold_start = _adapt_candidate(
            base, n, m, sparsity, seed, newfeat, holdouts, old_cols, egen)
        # reach (FVU_new<THRESH) and clean-reach (also forgetting<DELTA) per checkpoint
        reached_at = {s: int(fn < THRESH) for s, fn, fo in traj}
        clean_at = {s: int(fn < THRESH and (fo - fold_start) < DELTA) for s, fn, fo in traj}
        rows.append({
            "n_over_m": n // m, "sparsity": sparsity, "condition": condition,
            "seed": seed, "newfeat": newfeat,
            "grad_norm_init": grad_init,
            "fvu_new_end": traj[-1][1],
            "fvu_old_start": fold_start, "fvu_old_end": traj[-1][2],
            "forgetting": traj[-1][2] - fold_start,
            "reach_traj": ";".join(f"{s}:{reached_at[s]}" for s, _, _ in traj),
            "clean_traj": ";".join(f"{s}:{clean_at[s]}" for s, _, _ in traj),
        })
    return rows


def _worker(job):
    torch.set_num_threads(1)
    n, m, S, cond, seed = job
    return run_cell(n, m, S, cond, seed, CFG)


def main():
    cells = [(2, 0.9), (4, 0.9), (8, 0.9)]
    seeds = (0, 1, 2, 3, 4)
    jobs = [(nm * CFG.m, CFG.m, S, cond, seed)
            for (nm, S) in cells for cond in CFG.conditions for seed in seeds]
    print(f"Innovability runs: {len(jobs)} (cells={cells}, conds={CFG.conditions}, "
          f"seeds={seeds}, battery={BATTERY})")
    all_rows = []
    workers = min(4, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, rows in enumerate(ex.map(_worker, jobs), 1):
            all_rows.extend(rows)
            r0 = rows[0]
            reached = sum(int(rw["fvu_new_end"] < THRESH) for rw in rows)
            clean = sum(int(rw["fvu_new_end"] < THRESH and rw["forgetting"] < DELTA)
                        for rw in rows)
            gmean = sum(rw["grad_norm_init"] for rw in rows) / len(rows)
            print(f"[{i:2d}/{len(jobs)}] n/m={r0['n_over_m']} {r0['condition']:9s} "
                  f"seed={r0['seed']} reached@end={reached}/{BATTERY} "
                  f"clean@end={clean}/{BATTERY} grad0={gmean:.3f}", flush=True)
    with open("results/innovability.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    print(f"Wrote results/innovability.csv ({len(all_rows)} candidate-runs)")


if __name__ == "__main__":
    main()
