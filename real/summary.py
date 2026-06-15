"""Summarise the realer-substrate study -> results/tx_summary.md + results/tx.png.

Tests whether the toy verdicts survive attention + depth on a multi-skill modular-
arithmetic transformer:
  - robustness sanity: did the regimes actually diverge?
  - lesion: is editability (low collateral on knockout) conferred by a regime, and does
    op-embedding interference predict collateral?
  - innovability: when a new skill is bolted on with no rehearsal, which basin forgets
    the old skills least while still acquiring the new one?
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONDS = ["baseline", "input_at", "lat"]
les = pd.read_csv("results/tx_lesion.csv")
rob = pd.read_csv("results/tx_robustness.csv")
inn = pd.read_csv("results/tx_innovability.csv")


def pear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


L = ["# Realer substrate — tiny transformer, three regimes — summary", "",
     "Multi-skill modular arithmetic (`a op b =`, 6 operations), 2-layer transformer. "
     "Shared clean pretrain → condition-specific fine-tune (LAT/input-AT as the "
     "fine-tuning methods they are in practice), eps=0.05. Then the two capability-"
     "modification probes. Seeds = "
     f"{sorted(rob.seed.unique().tolist())}.", ""]

# ---- robustness sanity ----
g = rob.groupby("condition").agg(clean=("clean_acc", "mean"),
                                 adv_emb=("adv_acc_emb", "mean"),
                                 adv_resid=("adv_acc_resid", "mean"))
L += ["## Robustness sanity (did the regimes diverge?)", "",
      "Accuracy under PGD at the input-embedding site and the residual (LAT) site. "
      "Higher = more robust there.", "",
      "| condition | clean | adv (emb/input site) | adv (resid/latent site) |",
      "|---|---:|---:|---:|"]
for c in CONDS:
    if c in g.index:
        L.append(f"| {c} | {g.loc[c,'clean']:.2f} | {g.loc[c,'adv_emb']:.2f} | "
                 f"{g.loc[c,'adv_resid']:.2f} |")
L.append("")

# ---- lesion ----
gen = les[les.self_drop >= 0.3]                       # genuine knockouts (capability removed)
L += ["## Lesion — editability (collateral of an op knockout; lower = more surgical)", "",
      "Capability = an operation, knocked out at its op-token embedding. Self_drop is the "
      "accuracy lost on the knocked-out op (the edit landed); collateral is the mean "
      "accuracy lost on the OTHER ops.", "",
      "| condition | mean collateral | mean self_drop | r(collateral, I_i) genuine |",
      "|---|---:|---:|---:|"]
for c in CONDS:
    s = les[les.condition == c]; sg = gen[gen.condition == c]
    L.append(f"| {c} | {s.collateral.mean():.4f} | {s.self_drop.mean():.2f} | "
             f"{pear(sg.collateral, sg.I_i):+.2f} |")
L.append("")
# paired: is LAT / input_at more editable than baseline, by seed (mean collateral)?
sc = les.groupby(["condition", "seed"]).collateral.mean().unstack(0)
if all(c in sc.columns for c in CONDS):
    lb = int((sc["lat"] < sc["baseline"]).sum()); ib = int((sc["input_at"] < sc["baseline"]).sum())
    li = int((sc["lat"] < sc["input_at"]).sum()); ns = len(sc)
    L += [f"Paired by seed: input-AT more editable than baseline {ib}/{ns}; "
          f"LAT more editable than baseline {lb}/{ns}; LAT more editable than input-AT "
          f"{li}/{ns}.", ""]

# ---- innovability ----
gi = inn.groupby("condition").agg(new=("new_acc_end", "mean"),
                                  forget=("forgetting", "mean"),
                                  old_end=("old_acc_end", "mean"))
L += ["## Innovability — bolt on a new skill with NO rehearsal", "",
      "Adapt a held-out op (clean SGD, new op only) into each regime's basin. "
      "`new_acc_end` = acquired the new skill; `forgetting` = old-skill accuracy lost "
      "(lower = the basin integrates the new skill more cleanly).", "",
      "| condition | new_acc_end | old_acc_end | forgetting |", "|---|---:|---:|---:|"]
for c in CONDS:
    if c in gi.index:
        L.append(f"| {c} | {gi.loc[c,'new']:.2f} | {gi.loc[c,'old_end']:.2f} | "
                 f"{gi.loc[c,'forget']:+.3f} |")
L.append("")
# paired by (holdout, seed): does the robust basin forget less?
piv = inn.pivot_table(index=["holdout", "seed"], columns="condition", values="forgetting")
if all(c in piv.columns for c in CONDS):
    n = len(piv)
    ib = int((piv["input_at"] < piv["baseline"]).sum())
    lb = int((piv["lat"] < piv["baseline"]).sum())
    li = int((piv["lat"] < piv["input_at"]).sum())
    L += [f"Paired by (held-out op, seed), n={n}: input-AT forgets less than baseline "
          f"{ib}/{n}; LAT forgets less than baseline {lb}/{n}; LAT forgets less than "
          f"input-AT {li}/{n}.", ""]

# ---- verdict scaffold ----
order_les = " < ".join(f"{c}({les[les.condition==c].collateral.mean():.3f})"
                       for c in les.groupby('condition').collateral.mean().sort_values().index)
order_for = " < ".join(f"{c}({inn[inn.condition==c].forgetting.mean():+.3f})"
                       for c in inn.groupby('condition').forgetting.mean().sort_values().index)
L += ["## Verdict (does the toy story survive attention + depth?)", "",
      f"- Editability (lesion collateral, lower better): {order_les}.",
      f"- Innovability (forgetting, lower better): {order_for}.",
      "", "Compare to the toy: input-AT won BOTH directions of capability modification "
      "there. The table above says whether that holds with attention + depth.", ""]

open("results/tx_summary.md", "w").write("\n".join(L))

# ---- plot: editability bars + innovability forgetting bars ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
colors = {"baseline": "#888", "input_at": "#d62728", "lat": "#1f77b4"}
ax = axes[0]
vals = [les[les.condition == c].collateral.mean() for c in CONDS]
ax.bar(CONDS, vals, color=[colors[c] for c in CONDS])
ax.set_ylabel("mean collateral of knockout"); ax.set_title("Editability (lower = more surgical)")
ax = axes[1]
vals = [inn[inn.condition == c].forgetting.mean() for c in CONDS]
ax.bar(CONDS, vals, color=[colors[c] for c in CONDS])
ax.set_ylabel("old-skill forgetting"); ax.set_title("Innovability: forgetting when bolting\non a new skill (lower = cleaner)")
fig.suptitle("Realer substrate (transformer): capability modification by regime")
fig.tight_layout()
fig.savefig("results/tx.png", dpi=120)

print("\n".join(L))
print("\nWrote results/tx_summary.md and results/tx.png")
