"""Summarise results/followups.csv into results/followups_summary.md (CRITIQUE P2)."""
import pandas as pd

df = pd.read_csv("results/followups.csv")
lines = ["# Follow-up experiments — summary", "",
         "Generated from `results/followups.csv` (see `followups.py`, `CRITIQUE.md`).",
         "Cells: low-capacity n/m=2 vs high-capacity n/m=8, S=0.9. Seeds averaged.", ""]

# ---- experiment 1: eps sweep (LAT) ----
lines += ["## 1. epsilon dose-response (LAT)", "",
          "Mean over seeds; baseline shown as eps=0.00 reference.", ""]
for nm in sorted(df.n_over_m.unique()):
    base = df[(df.n_over_m == nm) & (df.condition == "baseline")]
    lat = df[(df.n_over_m == nm) & (df.condition == "lat")]
    lines.append(f"**n/m = {nm}**")
    lines.append("")
    lines.append("| eps | interference | mean_D | r_set2 (basin) |")
    lines.append("|----:|---:|---:|---:|")
    b = base.mean(numeric_only=True)
    lines.append(f"| 0.00 (base) | {b.interference:.3f} | {b.mean_D:.3f} | {b.r_set2_mean:.3f} |")
    for eps in sorted(lat.eps_rel.unique()):
        g = lat[lat.eps_rel == eps].mean(numeric_only=True)
        lines.append(f"| {eps:.2f} | {g.interference:.3f} | {g.mean_D:.3f} | {g.r_set2_mean:.3f} |")
    lines.append("")

# ---- experiment 2: targeted vs untargeted LAT ----
lines += ["## 2. targeted vs untargeted LAT (eps=0.10)", "",
          "`D_target` = ground-truth dimensionality of the corrupted concept (feature 0);",
          "higher = that concept is cleaner / more its own direction.", "",
          "| n/m | condition | interference | mean_D | D_target | r_set2 | r_set2_target |",
          "|----:|---|---:|---:|---:|---:|---:|"]
for nm in sorted(df.n_over_m.unique()):
    for cond in ["baseline", "lat", "lat_targeted"]:
        g = df[(df.n_over_m == nm) & (df.condition == cond) &
               ((df.eps_rel == 0.10) | (cond == "baseline"))].mean(numeric_only=True)
        lines.append(f"| {nm} | {cond} | {g.interference:.3f} | {g.mean_D:.3f} | "
                     f"{g.D_target:.3f} | {g.r_set2_mean:.3f} | {g.r_set2_target:.3f} |")
lines.append("")

with open("results/followups_summary.md", "w") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
