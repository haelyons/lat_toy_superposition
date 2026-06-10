"""Transformer control for the selectivity ablation.

The toy says untargeted LAT concentrates concepts in proportion to their importance/
frequency. This ports the test to the transformer: train with a SKEWED op-frequency
distribution (geometric), then ask whether the LAT-induced change in each op's
representation concentration tracks that op's frequency. If frequent ops concentrate more
under LAT (and rare ops don't), the "general LAT, selective effect" mechanism holds with
attention + depth — and a uniform-frequency model should show no such grading.

Concentration of op i = effective dimensionality (participation ratio, lower = more
concentrated) of the op's activation-difference cloud at the LAT perturbation site
(residual after `resid_layer`, op-token position).

Writes results/tx_selectivity.csv.
"""
import copy
import csv
import os
from concurrent.futures import ProcessPoolExecutor

import torch

from real.task import ModArithTask
from real.train import TXConfig, pretrain_clean, finetune, accuracy
from real.probes import OP_POS

CFG = TXConfig()
FREQ_DECAY = 0.6                    # geometric op-frequency skew (op 0 most frequent)


def skewed_split(task, seed):
    """Full table, but each op's rows replicated ∝ geometric frequency weight, so uniform
    sampling yields a skewed op-frequency distribution. Returns (split, freq_weight[op])."""
    base = task.make_split(frac=1.0, seed=seed)["train"]
    X, Y, OI = base
    factors = [max(1, round(13 * FREQ_DECAY ** o)) for o in range(task.K)]
    xs, ys, ois = [], [], []
    for o in range(task.K):
        m = OI == o
        for _ in range(factors[o]):
            xs.append(X[m]); ys.append(Y[m]); ois.append(OI[m])
    split = (torch.cat(xs), torch.cat(ys), torch.cat(ois))
    tot = sum(factors)
    return split, [f / tot for f in factors]


@torch.no_grad()
def _resid_at_oppos(model, tokens, layer):
    x = model.embed(tokens)
    mask = model._mask[:tokens.shape[1], :tokens.shape[1]]
    for i, blk in enumerate(model.blocks):
        x = blk(x, mask)
        if i == layer - 1:
            break
    return x[:, OP_POS, :]


@torch.no_grad()
def op_concentration(model, eval_split, task, cfg):
    """Per-op participation ratio of the activation-difference cloud at the LAT site."""
    X, _, OI = eval_split
    R = _resid_at_oppos(model, X, cfg.resid_layer)
    out = {}
    for o in range(task.K):
        m = OI == o
        if m.sum() < 8 or (~m).sum() < 8:
            out[o] = float("nan"); continue
        cloud = R[m] - R[~m].mean(0)
        lam = torch.linalg.svdvals(cloud) ** 2
        out[o] = float(((lam.sum() ** 2) / (lam ** 2).sum().clamp_min(1e-12)).item())
    return out


@torch.no_grad()
def op_embed_norms(model, task):
    """Per-op token-embedding norm — the transformer analogue of the toy's w_norm
    (reliance/magnitude), which is where the toy's importance-grading actually lived."""
    return {o: float(model.tok.weight[task.op_token(o)].norm().item()) for o in range(task.K)}


def _worker(seed):
    torch.set_num_threads(1)
    task = ModArithTask(p=CFG.p)
    split, freq = skewed_split(task, seed)
    eval_split = task.make_split(frac=1.0, seed=seed)["train"]   # balanced eval
    base = pretrain_clean(task, split, seed, CFG)
    rows = []
    conc, enorm = {}, {}
    for cond in ("baseline", "lat"):
        m = finetune(copy.deepcopy(base), task, split, cond, seed, CFG)
        conc[cond] = op_concentration(m, eval_split, task, CFG)
        enorm[cond] = op_embed_norms(m, task)
    for o in range(task.K):
        rows.append({"seed": seed, "op": task.ops[o], "op_idx": o,
                     "freq": round(freq[o], 4),
                     "pr_baseline": conc["baseline"][o], "pr_lat": conc["lat"][o],
                     "d_pr_lat": conc["lat"][o] - conc["baseline"][o],
                     "embnorm_baseline": enorm["baseline"][o], "embnorm_lat": enorm["lat"][o],
                     "d_embnorm_lat": enorm["lat"][o] - enorm["baseline"][o]})
    return rows


def main():
    seeds = list(CFG.seeds)
    print(f"TX selectivity (skewed op-freq, decay={FREQ_DECAY}): {len(seeds)} seeds")
    all_rows = []
    with ProcessPoolExecutor(max_workers=min(3, os.cpu_count() or 1)) as ex:
        for rows in ex.map(_worker, seeds):
            all_rows.extend(rows)
            for r in rows:
                print(f"  seed{r['seed']} {r['op']:4s} freq={r['freq']:.3f} "
                      f"pr {r['pr_baseline']:.2f}->{r['pr_lat']:.2f} Δ={r['d_pr_lat']:+.2f}",
                      flush=True)
    with open("results/tx_selectivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    print("Wrote results/tx_selectivity.csv")


if __name__ == "__main__":
    main()
