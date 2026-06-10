"""Summarise the innovability probe -> results/innovability_summary.md + a plot.

Headline question: does LAT's robust basin enable VIABLE innovation -- reaching new
concepts WITHOUT breaking old ones -- even though it is stiffer (slower) per target?

  raw innovability   = fraction of the battery with new-concept FVU<THRESH at a budget.
  clean innovability  = fraction also keeping old-concept FVU rise < DELTA (viable).

We report both as a function of SGD budget, the init gradient norm (direct stiffness),
and paired LAT-vs-baseline / LAT-vs-input_at win counts on clean innovability.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from innovability import CHECKPOINTS, THRESH, DELTA, BATTERY

df = pd.read_csv("results/innovability.csv")
CONDS = ["baseline", "input_at", "lat"]


def _parse(traj):
    return {int(p.split(":")[0]): int(p.split(":")[1]) for p in traj.split(";")}


# expand trajectories: per candidate-run, a dict step->{0,1} for reach and clean
df["_reach"] = df["reach_traj"].map(_parse)
df["_clean"] = df["clean_traj"].map(_parse)


def curve(sub, key):
    """Mean fraction (over candidate-runs) at each checkpoint."""
    return [np.mean([d[s] for d in sub[key]]) for s in CHECKPOINTS]


L = ["# Innovability probe — summary", "",
     "Wagner's evolvability = **innovability**: breadth of new concepts reachable "
     "*without sacrificing existing function*. `evolvability.py` found LAT forgets less "
     "but is stiffer; here we test the joint, evolutionarily honest criterion (innovate "
     "while remaining viable) across a battery of "
     f"{BATTERY} held-out niches, 5 seeds, cells n/m∈{{2,4,8}}, S=0.9.", "",
     f"**reached** = new-concept FVU<{THRESH}. **clean** = reached AND old-concept FVU "
     f"rise<{DELTA} (function preserved). Budget = {CHECKPOINTS[-1]} SGD steps.", ""]

# ---- direct stiffness: init gradient norm on the new concept ----
g = df.groupby(["n_over_m", "condition"]).grad_norm_init.mean().reset_index()
L += ["## Stiffness (direct): mean ‖∇‖ on the new concept at adaptation step 0", "",
      "Smaller = stiffer basin (slower descent), the mechanism behind slower acquisition.", "",
      "| n/m | baseline | input_at | lat |", "|----:|---:|---:|---:|"]
for nm in sorted(g.n_over_m.unique()):
    row = {r.condition: r.grad_norm_init for _, r in g[g.n_over_m == nm].iterrows()}
    L.append(f"| {nm} | {row.get('baseline',0):.3f} | {row.get('input_at',0):.3f} | "
             f"{row.get('lat',0):.3f} |")
L.append("")

# ---- raw vs clean innovability at a mid and full budget ----
mid = 150
L += ["## Innovability vs budget (fraction of battery)", "",
      f"raw = reached; clean = reached & viable. Shown at an early budget ({mid} steps) "
      f"and full budget ({CHECKPOINTS[-1]} steps).", "",
      "| n/m | condition | raw@%d | raw@%d | clean@%d | clean@%d |"
      % (mid, CHECKPOINTS[-1], mid, CHECKPOINTS[-1]),
      "|----:|---|---:|---:|---:|---:|"]
for nm in sorted(df.n_over_m.unique()):
    for cond in CONDS:
        sub = df[(df.n_over_m == nm) & (df.condition == cond)]
        rc = curve(sub, "_reach"); cc = curve(sub, "_clean")
        i_mid, i_end = CHECKPOINTS.index(mid), len(CHECKPOINTS) - 1
        L.append(f"| {nm} | {cond} | {rc[i_mid]:.2f} | {rc[i_end]:.2f} | "
                 f"{cc[i_mid]:.2f} | {cc[i_end]:.2f} |")
L.append("")

# ---- area under clean-innovability curve (single scalar; budget-integrated) ----
def auc(sub, key):
    y = curve(sub, key)
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trapz(y, CHECKPOINTS) / (CHECKPOINTS[-1] - CHECKPOINTS[0]))

L += ["## Budget-integrated clean innovability (area under clean curve, 0–1)", "",
      "One scalar per condition: higher = more viable novelty accessible across budgets.", "",
      "| n/m | baseline | input_at | lat |", "|----:|---:|---:|---:|"]
for nm in sorted(df.n_over_m.unique()):
    vals = {c: auc(df[(df.n_over_m == nm) & (df.condition == c)], "_clean") for c in CONDS}
    L.append(f"| {nm} | {vals['baseline']:.3f} | {vals['input_at']:.3f} | {vals['lat']:.3f} |")
L.append("")

# ---- paired win-counts (per seed, averaged over the battery) on clean innovability ----
# Per (seed): mean clean@end and AUC over its battery, then count seed-cells where LAT wins.
def per_seed_metric(sub, key, at_end=True):
    out = {}
    for seed, s2 in sub.groupby("seed"):
        if at_end:
            out[seed] = np.mean([d[CHECKPOINTS[-1]] for d in s2[key]])
        else:
            out[seed] = auc(s2, key)
    return out

L += ["## Paired win-counts (per seed, battery-averaged): clean innovability", "",
      "Counts seeds where LAT ≥ comparator on budget-integrated clean innovability "
      "(ties to LAT broken as wins only if strictly ≥; reported as wins/seeds).", "",
      "| n/m | LAT ≥ baseline | LAT ≥ input_at |", "|----:|---:|---:|"]
tot_b = tot_i = tot_n = 0
for nm in sorted(df.n_over_m.unique()):
    sub = df[df.n_over_m == nm]
    a_lat = per_seed_metric(sub[sub.condition == "lat"], "_clean", at_end=False)
    a_base = per_seed_metric(sub[sub.condition == "baseline"], "_clean", at_end=False)
    a_inp = per_seed_metric(sub[sub.condition == "input_at"], "_clean", at_end=False)
    seeds = sorted(a_lat)
    wb = sum(a_lat[s] >= a_base[s] for s in seeds)
    wi = sum(a_lat[s] >= a_inp[s] for s in seeds)
    tot_b += wb; tot_i += wi; tot_n += len(seeds)
    L.append(f"| {nm} | {wb}/{len(seeds)} | {wi}/{len(seeds)} |")
L += ["", f"**Totals:** LAT ≥ baseline in {tot_b}/{tot_n} seed-cells; "
      f"LAT ≥ input_at in {tot_i}/{tot_n}.", ""]

open("results/innovability_summary.md", "w").write("\n".join(L))

# ---- plot: raw (dashed) vs clean (solid) innovability curves per cell ----
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
colors = {"baseline": "#888", "input_at": "#d62728", "lat": "#1f77b4"}
for ax, nm in zip(axes, sorted(df.n_over_m.unique())):
    for cond in CONDS:
        sub = df[(df.n_over_m == nm) & (df.condition == cond)]
        ax.plot(CHECKPOINTS, curve(sub, "_reach"), "--", color=colors[cond], alpha=0.6)
        ax.plot(CHECKPOINTS, curve(sub, "_clean"), "-", color=colors[cond], lw=2,
                label=cond)
    ax.set_title(f"n/m = {nm}")
    ax.set_xlabel("adaptation steps")
    ax.set_xscale("symlog")
axes[0].set_ylabel("fraction of battery")
axes[0].legend(title="solid=clean, dashed=raw", fontsize=8)
fig.suptitle("Innovability: viable new-concept acquisition (clean) vs raw reach")
fig.tight_layout()
fig.savefig("results/innovability.png", dpi=120)

print("\n".join(L))
print("\nWrote results/innovability_summary.md and results/innovability.png")
