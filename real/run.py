"""Orchestrate the realer-substrate study: tiny transformer, three regimes, then the
two capability-modification probes.

Phase 0 (sanity): confirm the regimes actually differ — clean accuracy (matched) and
  adversarial accuracy under emb-PGD (input site) and resid-PGD (latent site).
Phase 1 (lesion): per seed, clean-pretrain on all ops, fine-tune each condition, knock
  out each op's residual direction, record self_drop / collateral / interference.
Phase 2 (innovability): per held-out op, build condition-specific basins (pretrain on the
  rest, fine-tune each regime), then adapt the new op with identical clean SGD; record
  acquisition and forgetting.

Writes results/tx_lesion.csv, results/tx_innovability.csv, results/tx_robustness.csv.
"""
import csv
import copy
import os
from concurrent.futures import ProcessPoolExecutor

import torch

from real.task import ModArithTask
from real.train import TXConfig, pretrain_clean, finetune, accuracy, _pgd_perturb, _ce
from real.probes import lesion, adapt_new_op, _subset

CFG = TXConfig()
HOLDOUTS = [1, 2, 4]          # sub, mul, max — additive / multiplicative / order skills


def adv_accuracy(model, split, cfg, site):
    X, Y, _ = split
    d = _pgd_perturb(model, X, Y, cfg, site)
    with torch.no_grad():
        if site == "emb":
            logits = model(X, emb_delta=d)
        else:
            logits = model(X, resid_delta=d, resid_layer=cfg.resid_layer)
        return (logits[:, -1, :].argmax(-1) == Y).float().mean().item()


def _phase1_worker(seed):
    torch.set_num_threads(1)
    task = ModArithTask(p=CFG.p)
    split = task.make_split(frac=1.0, seed=seed)["train"]
    base = pretrain_clean(task, split, seed, CFG)
    lesion_rows, robust_rows = [], []
    for cond in CFG.conditions:
        m = finetune(copy.deepcopy(base), task, split, cond, seed, CFG)
        for r in lesion(m, split, task):
            r.update({"seed": seed, "condition": cond}); lesion_rows.append(r)
        robust_rows.append({"seed": seed, "condition": cond,
                            "clean_acc": accuracy(m, split, task),
                            "adv_acc_emb": adv_accuracy(m, split, CFG, "emb"),
                            "adv_acc_resid": adv_accuracy(m, split, CFG, "resid")})
    return lesion_rows, robust_rows


def _phase2_worker(job):
    torch.set_num_threads(1)
    holdout, seed = job
    task = ModArithTask(p=CFG.p)
    keep = [o for o in range(task.K) if o != holdout]
    sub = _subset(task.make_split(frac=1.0, seed=seed)["train"], keep)   # pretrain on rest
    base = pretrain_clean(task, sub, seed, CFG)
    rows = []
    for cond in CFG.conditions:
        basin = finetune(copy.deepcopy(base), task, sub, cond, seed, CFG,
                         steps=600)                          # imprint the regime's basin
        traj = adapt_new_op(basin, task, CFG, holdout, seed, steps=600, log_every=60)
        new_end, old_start, old_end = traj[-1][1], traj[0][2], traj[-1][2]
        thr = next((s for s, na, oa in traj if na >= 0.9), -1)
        rows.append({"holdout": task.ops[holdout], "holdout_idx": holdout, "seed": seed,
                     "condition": cond, "new_acc_end": new_end,
                     "old_acc_start": old_start, "old_acc_end": old_end,
                     "forgetting": old_start - old_end, "steps_to_new90": thr,
                     "traj": ";".join(f"{s}:{na:.2f}:{oa:.2f}" for s, na, oa in traj)})
    return rows


def main():
    os.makedirs("results", exist_ok=True)
    workers = min(3, os.cpu_count() or 1)
    seeds = list(CFG.seeds)

    print(f"== Phase 1 (lesion) + sanity: {len(seeds)} seeds ==", flush=True)
    lesion_all, robust_all = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for lr, rr in ex.map(_phase1_worker, seeds):
            lesion_all.extend(lr); robust_all.extend(rr)
            for r in rr:
                print(f"  seed{r['seed']} {r['condition']:9s} clean={r['clean_acc']:.2f} "
                      f"adv_emb={r['adv_acc_emb']:.2f} adv_resid={r['adv_acc_resid']:.2f}",
                      flush=True)

    print(f"== Phase 2 (innovability): holdouts={HOLDOUTS} x {len(seeds)} seeds ==", flush=True)
    jobs = [(h, s) for h in HOLDOUTS for s in seeds]
    innov_all = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for rows in ex.map(_phase2_worker, jobs):
            innov_all.extend(rows)
            for r in rows:
                print(f"  holdout={r['holdout']:4s} seed{r['seed']} {r['condition']:9s} "
                      f"new={r['new_acc_end']:.2f} forget={r['forgetting']:+.3f} "
                      f"steps2new90={r['steps_to_new90']}", flush=True)

    for name, rowset in [("tx_lesion", lesion_all), ("tx_robustness", robust_all),
                         ("tx_innovability", innov_all)]:
        with open(f"results/{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rowset[0].keys()))
            w.writeheader(); w.writerows(rowset)
    print("Wrote results/tx_{lesion,robustness,innovability}.csv")


if __name__ == "__main__":
    main()
