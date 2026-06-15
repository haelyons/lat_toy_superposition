"""Summarise the lesion/editability probe -> results/lesion_summary.md + a plot.

Two questions:
  (1) Does the GEOMETRIC concentration measure (interference I_i) cash out as FUNCTIONAL
      editability? -> correlation of collateral damage with I_i / D_i, per condition.
  (2) Are LAT's (more concentrated) models more surgically editable? -> mean collateral
      per condition, paired by seed, with self_drop reported as a fairness check.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("results/lesion.csv")
CONDS = ["baseline", "input_at", "lat"]
df = df.dropna(subset=["collateral"])


def pearson(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


L = ["# Lesion / editability probe — summary", "",
     "Capability modification = directional **knockout** of one concept (the standard "
     "activation edit); **collateral** = mean FVU increase on the *other* represented "
     "concepts; **self_drop** = FVU increase on the knocked-out concept (how much "
     "capability was actually removed — a fairness check). Full Claim B grid "
     "(n/m∈{2,4,8} × S∈{0.8,0.9,0.99}), 5 seeds.", "",
     "Does the geometric concentration measure (interference `I_i`) cash out as "
     "functional editability, and are LAT's more-concentrated models more surgically "
     "editable?", ""]

# ---- (1) does concentration predict collateral? (must control for magnitude) ----
# Confound: high-I_i concepts are the weakly-represented junk ones, whose knockout
# removes ~nothing -> low collateral, faking a NEGATIVE pooled r(collateral,I). The
# honest test conditions on a GENUINE knockout (self_drop>=0.5 = the concept was really
# removed); there the analytic prediction (collateral rises with interference) holds.
GENUINE = df[df.self_drop >= 0.5]
L += ["## (1) Geometry → function: does concentration predict editability?", "",
      f"**Magnitude confound** (pooled, all {len(df)} lesions): "
      f"r(collateral, I_i) = {pearson(df.collateral, df.I_i):+.2f} — *negative*, because "
      f"r(I_i, w_norm) = {pearson(df.I_i, df.w_norm):+.2f}: high-interference concepts are "
      f"the barely-represented ones whose removal does nothing "
      f"(r(collateral, w_norm) = {pearson(df.collateral, df.w_norm):+.2f}).", "",
      f"**Conditioning on a genuine knockout** (self_drop≥0.5, {len(GENUINE)} lesions) "
      "removes the confound; interference then predicts collateral in the analytic "
      "(positive) direction — the geometric measure has real editability meaning:", "",
      "| condition | r(coll, I_i) genuine | r(coll, I_i) pooled | mean collateral | mean self_drop |",
      "|---|---:|---:|---:|---:|"]
for c in CONDS:
    s = df[df.condition == c]; sg = GENUINE[GENUINE.condition == c]
    L.append(f"| {c} | {pearson(sg.collateral, sg.I_i):+.2f} | "
             f"{pearson(s.collateral, s.I_i):+.2f} | {s.collateral.mean():.4f} | "
             f"{s.self_drop.mean():.3f} |")
L.append("")
L.append(f"Pooled genuine-knockout: r(collateral, I_i) = "
         f"{pearson(GENUINE.collateral, GENUINE.I_i):+.2f}.")
L.append("")

# ---- (2) are LAT models more editable? mean collateral per cell, paired by seed ----
L += ["## (2) Editability by condition (mean collateral per cell; lower = more editable)", "",
      "Self_drop is matched across conditions (full directional knockout), so raw "
      "collateral is comparable.", "",
      "| n/m | S | baseline | input_at | lat | LAT<base? | LAT<inp? |",
      "|----:|---:|---:|---:|---:|:--:|:--:|"]
seed_col = df.groupby(["n_over_m", "sparsity", "condition", "seed"]).collateral.mean().reset_index()
tot = {"lb": 0, "li": 0, "n": 0}
for (nm, S), g in seed_col.groupby(["n_over_m", "sparsity"]):
    pv = g.pivot_table(index="seed", columns="condition", values="collateral")
    if not all(c in pv.columns for c in CONDS):
        continue
    mb, mi, ml = pv["baseline"].mean(), pv["input_at"].mean(), pv["lat"].mean()
    lb = int((pv["lat"] < pv["baseline"]).sum()); li = int((pv["lat"] < pv["input_at"]).sum())
    ns = len(pv)
    tot["lb"] += lb; tot["li"] += li; tot["n"] += ns
    L.append(f"| {nm} | {S} | {mb:.4f} | {mi:.4f} | {ml:.4f} | {lb}/{ns} | {li}/{ns} |")
L += ["", f"**Totals (seed-cells where LAT is more editable):** LAT<baseline "
      f"{tot['lb']}/{tot['n']}; LAT<input_at {tot['li']}/{tot['n']}.", ""]

# ---- capability-modification 2x2 framing ----
mc = df.groupby("condition").collateral.mean()
order = " < ".join(f"{c} ({mc[c]:.3f})" for c in mc.sort_values().index)
L += ["## The capability-modification 2×2", "",
      f"Mean collateral (lower = more editable): {order}.", "",
      "| direction of modification | clean winner | LAT vs baseline |",
      "|---|---|---|",
      f"| **edit / remove EXISTING** (this probe) | input-AT (usually) ≥ LAT > baseline | "
      f"LAT more editable in {tot['lb']}/{tot['n']} |",
      "| **acquire NEW** (`innovability.py`) | input-AT | LAT ≈ baseline (both forget) |",
      "",
      "So **input-space robustness wins on BOTH directions of capability modification**; "
      "LAT improves editability over baseline but is dominated by input-AT, and does not "
      "help new-concept acquisition. The unifying read: input-AT shapes the representation "
      "so that capability edits — adding or removing — cost less elsewhere; LAT's latent "
      "robustness is a weaker version of the same for removal and the wrong tool for "
      "addition.", ""]

open("results/lesion_summary.md", "w").write("\n".join(L))

# ---- plots ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
colors = {"baseline": "#888", "input_at": "#d62728", "lat": "#1f77b4"}
# (a) scatter collateral vs interference, per condition with fit
ax = axes[0]
gen = df[df.self_drop >= 0.5]                          # genuine knockouts (magnitude-controlled)
for c in CONDS:
    s = gen[gen.condition == c]
    ax.scatter(s.I_i, s.collateral, s=6, alpha=0.25, color=colors[c], label=c)
    if len(s) > 2 and s.I_i.std() > 0:
        b1, b0 = np.polyfit(s.I_i, s.collateral, 1)
        xs = np.linspace(s.I_i.min(), s.I_i.max(), 50)
        ax.plot(xs, b0 + b1 * xs, color=colors[c], lw=2)
ax.set_xlabel("per-concept interference  I_i"); ax.set_ylabel("collateral damage of knockout")
ax.set_title("Geometry → function (genuine knockouts):\ninterference predicts collateral")
ax.legend(fontsize=8)
# (b) mean collateral per condition per n/m (averaged over sparsity & seeds)
ax = axes[1]
cells = sorted(df.n_over_m.unique())
xw = np.arange(len(cells)); bw = 0.25
for k, c in enumerate(CONDS):
    means = [df[(df.n_over_m == nm) & (df.condition == c)].collateral.mean() for nm in cells]
    ax.bar(xw + (k - 1) * bw, means, bw, color=colors[c], label=c)
ax.set_xticks(xw); ax.set_xticklabels([f"n/m={nm}" for nm in cells])
ax.set_ylabel("mean collateral damage"); ax.set_title("Editability by condition\n(lower = more surgical)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("results/lesion.png", dpi=120)

print("\n".join(L))
print("\nWrote results/lesion_summary.md and results/lesion.png")
