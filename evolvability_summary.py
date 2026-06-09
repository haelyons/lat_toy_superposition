"""Summarise the evolvability probe into results/evolvability_summary.md + a verdict.

Tests the Wagner prediction: LAT (wider/cleaner basin) is MORE evolvable -- a new
concept is more accessible (lower fvu_new_start), acquired at least as fast, and
integrated with less forgetting -- vs baseline and input-AT, paired by seed.
"""
import pandas as pd

ad = pd.read_csv("results/evolvability_adaptation.csv")
comp = pd.read_csv("results/evolvability_compositionality.csv")

L = ["# Evolvability probe — summary", "",
     "Wagner 'robustness enables evolvability' test (SPEC §10). Mean over seeds; "
     "paired LAT-vs-baseline by seed. `fvu_new_start` = zero-shot readability of a "
     "fresh new-concept direction (lower = more accessible niche in the basin); "
     "`forgetting` = old-concept FVU increase during adaptation (lower = cleaner "
     "integration); `steps_to_new_thresh` = SGD steps for new-concept FVU<0.20.", ""]

# ---- adaptation table ----
g = (ad.groupby(["n_over_m", "condition"])
       .agg(fvu_new_start=("fvu_new_start", "mean"),
            fvu_new_end=("fvu_new_end", "mean"),
            steps=("steps_to_new_thresh", "mean"),
            forgetting=("forgetting", "mean")).reset_index())
L += ["## Adaptation to a new concept", "",
      "| n/m | condition | fvu_new_start | steps→thresh | forgetting | fvu_new_end |",
      "|----:|---|---:|---:|---:|---:|"]
for _, r in g.iterrows():
    L.append(f"| {r.n_over_m} | {r.condition} | {r.fvu_new_start:.2f} | "
             f"{r.steps:.0f} | {r.forgetting:+.4f} | {r.fvu_new_end:.3f} |")
L.append("")

# ---- paired LAT vs baseline verdict ----
L += ["## LAT vs baseline (paired by seed)", "",
      "| n/m | Δ readability (LAT−base, <0 better) | Δ forgetting (LAT−base, <0 better) | Δ steps (LAT−base) |",
      "|----:|---:|---:|---:|"]
acc = {"read": 0, "forget": 0, "cells": 0}
for nm, df in ad.groupby("n_over_m"):
    piv_s = df.pivot_table(index="seed", columns="condition", values="fvu_new_start")
    piv_f = df.pivot_table(index="seed", columns="condition", values="forgetting")
    piv_t = df.pivot_table(index="seed", columns="condition", values="steps_to_new_thresh")
    dr = (piv_s["lat"] - piv_s["baseline"]).mean()
    dfg = (piv_f["lat"] - piv_f["baseline"]).mean()
    dt = (piv_t["lat"] - piv_t["baseline"]).mean()
    acc["cells"] += 1
    acc["read"] += int(dr < 0)
    acc["forget"] += int(dfg < 0)
    L.append(f"| {nm} | {dr:+.2f} | {dfg:+.4f} | {dt:+.0f} |")
L += ["",
      f"LAT improves new-concept readability (lower fvu_new_start) vs baseline in "
      f"{acc['read']}/{acc['cells']} cells; reduces forgetting in "
      f"{acc['forget']}/{acc['cells']} cells.", ""]

# ---- compositionality ----
cg = (comp.groupby(["n_over_m", "k_active", "condition"]).fvu_old.mean()
          .reset_index())
L += ["## Compositionality (FVU on novel dense combinations, k features co-active)", "",
      "Lower = better reconstruction of unseen combinations.", ""]
for nm in sorted(cg.n_over_m.unique()):
    sub = cg[cg.n_over_m == nm]
    ks = sorted(sub.k_active.unique())
    L.append(f"**n/m={nm}** — " + " | ".join(f"k={k}" for k in ks))
    L.append("| condition | " + " | ".join(f"k={k}" for k in ks) + " |")
    L.append("|---" * (len(ks) + 1) + "|")
    for cond in ["baseline", "input_at", "lat"]:
        vals = [sub[(sub.k_active == k) & (sub.condition == cond)].fvu_old.mean() for k in ks]
        L.append(f"| {cond} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    L.append("")

open("results/evolvability_summary.md", "w").write("\n".join(L))
print("\n".join(L))
