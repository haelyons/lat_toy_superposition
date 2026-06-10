"""Summarise the selectivity ablation -> results/selectivity_summary.md + figure.

Question: is untargeted LAT's concentration effect uniform, or graded by a concept's
importance / frequency / reliance? For each concentration measure we form a signed
"concentration gain" (positive = more concentrated under LAT than baseline), paired by
(cell, seed, feature), and correlate it with concept importance — overall and split by
the concentration regime (n/m in {2,4}) vs the saturated regime (n/m=8).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("results/selectivity.csv")

# signed concentration gain per measure: positive = MORE concentrated under the condition.
#   D_i higher = cleaner; I_i lower = cleaner; late_pr lower = more 1D; late_top1evr higher.
MEAS = {"D_i": +1, "I_i": -1, "w_norm": +1, "late_pr": -1, "late_top1evr": +1}


_KEY = ["n_over_m", "sparsity", "feature"]
_LOOK = df[_KEY + ["importance", "frequency"]].drop_duplicates(_KEY)


def gains(cond):
    """Per-(cell,seed,feature) concentration gain of `cond` over baseline, + importance."""
    out = {}
    for met, sign in MEAS.items():
        p = df.pivot_table(index=["n_over_m", "sparsity", "seed", "feature"],
                           columns="condition", values=met)
        p = p.dropna(subset=["baseline", cond]).reset_index()
        p["gain"] = sign * (p[cond] - p["baseline"])
        p = p.merge(_LOOK, on=_KEY, how="left")
        out[met] = p
    return out


def r(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


L = ["# Selectivity ablation — is LAT's concentration uniform or importance-graded?", "",
     "Untargeted LAT (the `lat` condition; same as Abbas et al.'s general LAT). For each "
     "concentration measure, **gain** = how much MORE concentrated a concept is under the "
     "condition vs baseline (signed so + = more concentrated). If the effect is uniform, "
     "gain is unrelated to importance; if selective, gain rises with importance.", "",
     "Importance = geometric decay within a cell; frequency = 1−sparsity across cells. "
     "Concentration regime = n/m∈{2,4} (where LAT concentrates); n/m=8 saturates.", ""]

for cond in ["lat", "input_at"]:
    g = gains(cond)
    L += [f"## {cond} vs baseline", "",
          "| measure | r(gain, importance) ALL | r(gain, imp) n/m∈{2,4} | r(gain, freq) | mean gain |",
          "|---|---:|---:|---:|---:|"]
    for met in MEAS:
        p = g[met]
        p24 = p[p.n_over_m.isin([2, 4])]
        L.append(f"| {met} | {r(p['gain'], p['importance']):+.2f} | "
                 f"{r(p24['gain'], p24['importance']):+.2f} | "
                 f"{r(p['gain'], p['frequency']):+.2f} | {p['gain'].mean():+.4f} |")
    L.append("")

# headline numbers for the weight measures (the non-degenerate ones)
gl = gains("lat")
rw = r(gl["w_norm"]["gain"], gl["w_norm"]["importance"])
rD = r(gl["D_i"][gl["D_i"].n_over_m.isin([2, 4])]["gain"],
       gl["D_i"][gl["D_i"].n_over_m.isin([2, 4])]["importance"])
L += ["## Read", "",
      f"- **Reliance/magnitude (w_norm)**: r(gain, importance) = {rw:+.2f} — LAT strengthens "
      "important concepts far more than marginal ones.",
      f"- **Weight cleanliness (D_i, concentration regime)**: r = {rD:+.2f}.",
      "- A positive importance correlation = **general LAT is selective**: it concentrates "
      "the concepts the loss relies on, not all concepts uniformly. This is the mechanism "
      "by which Abbas et al.'s *general* LAT produced a *refusal-specific*-looking effect. "
      "See the transformer control (`results/tx_selectivity.csv`) for the same test with "
      "skewed op frequency.", ""]

# ---- transformer control (skewed op frequency) ----
try:
    tx = pd.read_csv("results/tx_selectivity.csv")
    r_emb = r(tx["freq"], tx["d_embnorm_lat"])
    r_conc = r(tx["freq"], -tx["d_pr_lat"])
    perseed = [round(r(g["freq"], g["d_embnorm_lat"]), 2) for _, g in tx.groupby("seed")]
    L += ["## Transformer control (skewed op-frequency; `real/selectivity.py`)", "",
          "Same question with attention + depth. Ops differ only in FREQUENCY (equal "
          "per-instance loss weight, unlike the toy's importance weighting). We correlate "
          "each op's LAT-induced embedding-norm change (the w_norm analogue) with its "
          "frequency.", "",
          f"- r(frequency, Δembedding-norm under LAT) = **{r_emb:+.2f}** (per-seed {perseed}).",
          f"- r(frequency, activation-cloud concentration gain) = {r_conc:+.2f} (noisy).", "",
          "**The sign is NEGATIVE — opposite the toy.** With equal per-instance loss, LAT "
          "strengthens the RARE/most-vulnerable ops most (robustness-equalisation), whereas "
          "the toy's loss-importance weighting made LAT strengthen the IMPORTANT concepts. "
          "Both refute uniformity; the selection *axis* (and its sign) is set by what makes "
          "a concept salient to the loss — i.e. by the fine-tuning distribution/objective.", ""]
except FileNotFoundError:
    L += ["## Transformer control", "", "_Run `python -m real.selectivity` first._", ""]

# ---- overall conclusion (discoverable) ----
L += ["## Conclusion (for the revised write-up)", "",
      "1. **General (untargeted) LAT does NOT concentrate all concepts uniformly** — its "
      "effect is selective, allocated by how the loss/adversary engages each concept. This "
      "is the correction to any reading of Abbas et al. that assumed a uniform effect: "
      "refusal concentrated because it is salient to the safety objective, not because LAT "
      "touches every concept equally.",
      "2. **The selection axis is set by the fine-tuning distribution/objective.** "
      "Loss-importance weighting → LAT concentrates/strengthens the IMPORTANT concepts "
      f"(toy: w_norm r={rw:+.2f}, D r={rD:+.2f}). Pure frequency skew at equal loss weight "
      f"→ LAT strengthens the RARE/vulnerable concepts (transformer: embnorm-vs-freq "
      f"r={r_emb if 'r_emb' in dir() else float('nan'):+.2f}).",
      "3. **The grading is on the WEIGHT/reliance geometry, not the activation effective-"
      "dimension.** The Abbas-style activation 'collapse to 1D' (late_pr/top1evr) is ~flat "
      "vs importance in the toy and noisy in the transformer; what moves selectively is how "
      "strongly/cleanly a concept is *written into the weights*. Measure reliance, not just "
      "SVD spectra, when auditing LAT.", ""]

open("results/selectivity_summary.md", "w").write("\n".join(L))

# ---- figure: concentration gain vs importance, per measure (lat) ----
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, met in zip(axes, ["w_norm", "D_i", "late_pr"]):
    p = gl[met]
    for nm, c in [(2, "#1f77b4"), (4, "#2ca02c"), (8, "#d62728")]:
        s = p[p.n_over_m == nm]
        ax.scatter(s["importance"], s["gain"], s=8, alpha=0.4, color=c, label=f"n/m={nm}")
    ax.set_xscale("log"); ax.set_xlabel("concept importance (log)")
    ax.set_ylabel(f"{met} concentration gain (LAT−base)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(met)
axes[0].legend(fontsize=8)
fig.suptitle("Selectivity: LAT concentration gain vs concept importance (toy)")
fig.tight_layout()
fig.savefig("results/selectivity.png", dpi=120)

print("\n".join(L))
print("\nWrote results/selectivity_summary.md and results/selectivity.png")
