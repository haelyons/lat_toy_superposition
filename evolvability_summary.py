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

# ---- paired LAT vs baseline verdict (per-SEED counts -- means are seed-driven) ----
L += ["## LAT vs baseline (paired by seed; per-seed win counts, not cell means)", "",
      "Means can be dominated by a single seed, so we count seed-cells where LAT beats "
      "baseline. n_seeds=number of seeds per cell.", "",
      "| n/m | readability better (seeds) | forgetting lower (seeds) | adapts faster (seeds) |",
      "|----:|---:|---:|---:|"]
tot = {"read": 0, "forget": 0, "fast": 0, "N": 0}
for nm, df in ad.groupby("n_over_m"):
    ps = df.pivot_table(index="seed", columns="condition", values="fvu_new_start")
    pf = df.pivot_table(index="seed", columns="condition", values="forgetting")
    pt = df.pivot_table(index="seed", columns="condition", values="steps_to_new_thresh")
    ns = len(ps)
    nr = int(((ps["lat"] - ps["baseline"]) < 0).sum())
    nf = int(((pf["lat"] - pf["baseline"]) < 0).sum())
    nt = int(((pt["lat"] - pt["baseline"]) < 0).sum())
    tot["read"] += nr; tot["forget"] += nf; tot["fast"] += nt; tot["N"] += ns
    L.append(f"| {nm} | {nr}/{ns} | {nf}/{ns} | {nt}/{ns} |")
L += ["",
      f"**Across all {tot['N']} seed-cells:** LAT better new-concept readability in "
      f"{tot['read']}/{tot['N']}, lower forgetting in {tot['forget']}/{tot['N']}, "
      f"faster adaptation in {tot['fast']}/{tot['N']}.", ""]

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
