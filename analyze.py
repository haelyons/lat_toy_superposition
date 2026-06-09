"""Summary artifacts and the B/C verdict (SPEC 8).

Produces:
  (a) results/proxy_correlations.csv  - proxy vs ground-truth superposition (Claim C)
  (b) results/basin_comparison.csv    - r_i (trained vs unseen dirs) across conditions (Claim B)
  (c) results/concentration_vs_gt.csv - per-concept PR/SVD vs ground-truth superposition
  plots: results/*.png
  results/summary.json + printed verdict on Claims B and C.
"""
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, pearsonr


def load():
    runs = pd.read_csv("results/runs.csv")
    con = pd.read_csv("results/concepts.csv")
    return runs, con


def represented(con):
    """Concepts the model actually represents; the robust-region notion is only
    meaningful for these (unrepresented features have a degenerate, censored radius)."""
    return con[con["represented"] == 1].copy()


# ---------- (a) proxy vs ground-truth correlations (Claim C) ----------

def proxy_correlations(con):
    """Correlate each concept proxy against ground-truth dimensionality D_i,
    pooled and per (n_over_m, sparsity, condition) regime."""
    proxies = ["top1_evr", "pr", "r_set1", "r_set2"]
    rows = []

    def corr_block(df, label):
        gt = df["D_groundtruth"].to_numpy()
        for p in proxies:
            v = df[p].to_numpy()
            mask = np.isfinite(gt) & np.isfinite(v)
            if mask.sum() < 8 or np.std(v[mask]) < 1e-9:
                sp = pr = np.nan
            else:
                sp = spearmanr(gt[mask], v[mask]).correlation
                pr = pearsonr(gt[mask], v[mask])[0]
            rows.append({"regime": label, "proxy": p,
                         "spearman_vs_D": sp, "pearson_vs_D": pr, "nobs": int(mask.sum())})

    # proxies derived from the readout (r_set1/r_set2) are only meaningful for
    # represented concepts; evr/pr are defined for all. Use represented set for
    # the correlation so dead-feature censoring does not dominate.
    con = represented(con)
    corr_block(con, "ALL")
    for cond, df in con.groupby("condition"):
        corr_block(df, f"cond={cond}")
    for (nm, S), df in con.groupby(["n_over_m", "sparsity"]):
        corr_block(df, f"n/m={nm},S={S}")
    out = pd.DataFrame(rows)
    out.to_csv("results/proxy_correlations.csv", index=False)
    return out


# ---------- (b) robust-basin comparison (Claim B) ----------

def basin_comparison(con):
    """Mean +/- std r_set1 / r_set2 per (cell, condition), over represented concepts & seeds."""
    con = represented(con)
    g = (con.groupby(["n_over_m", "sparsity", "condition"])
            .agg(r_set1_mean=("r_set1", "mean"), r_set1_std=("r_set1", "std"),
                 r_set2_mean=("r_set2", "mean"), r_set2_std=("r_set2", "std"),
                 pr_mean=("pr", "mean"), pr_std=("pr", "std"),
                 sat_set2=("sat_set2", "mean"),
                 nobs=("r_set1", "count"))
            .reset_index())
    g.to_csv("results/basin_comparison.csv", index=False)
    return g


# ---------- (b2) weight-level concentration across conditions (Claim B, corrected) ----------

def weight_concentration(runs):
    """Does LAT concentrate the concept GEOMETRY (weights), paired by seed per cell?

    The per-concept latent-cloud proxy (top1_evr/pr) is measured on the linear
    early latent h1 = W1 x, where each feature's signal is structurally rank-1, so
    its apparent dimensionality is pure co-activation interference (set by sparsity)
    and cannot move with training. The quantity that CAN move is the ground-truth
    weight geometry: mean per-feature dimensionality D (higher = cleaner / less
    superposition) and off-diagonal interference (lower = cleaner). We compare LAT
    and input-AT to baseline, paired by seed, in every cell."""
    rows = []
    for (nm, S), df in runs.groupby(["n_over_m", "sparsity"]):
        pv_D = df.pivot_table(index="seed", columns="condition", values="mean_D")
        pv_I = df.pivot_table(index="seed", columns="condition", values="interference")
        if not all(c in pv_D.columns for c in ["baseline", "lat", "input_at"]):
            continue
        dD = pv_D["lat"] - pv_D["baseline"]
        dI = pv_I["lat"] - pv_I["baseline"]
        rows.append({
            "n_over_m": nm, "sparsity": S,
            "D_base": pv_D["baseline"].mean(), "D_lat": pv_D["lat"].mean(),
            "dD_lat_minus_base": dD.mean(), "dD_seeds_pos": int((dD > 0).sum()),
            "I_base": pv_I["baseline"].mean(), "I_lat": pv_I["lat"].mean(),
            "dI_lat_minus_base": dI.mean(), "dI_seeds_neg": int((dI < 0).sum()),
            "n_seeds": len(dD),
            "lat_concentrates": bool(dD.mean() > 0 and dI.mean() < 0),
        })
    out = pd.DataFrame(rows)
    out.to_csv("results/weight_concentration.csv", index=False)
    return out


# ---------- (c) concentration vs ground truth ----------

def concentration_vs_gt(con):
    g = (con.groupby(["n_over_m", "sparsity", "condition"])
            .agg(top1_evr_mean=("top1_evr", "mean"), pr_mean=("pr", "mean"),
                 D_mean=("D_groundtruth", "mean"), interference_proxy=("D_groundtruth", "std"))
            .reset_index())
    g.to_csv("results/concentration_vs_gt.csv", index=False)
    return g


# ---------- plots ----------

def plot_basin(con):
    con = represented(con)
    cells = sorted(con[["n_over_m", "sparsity"]].drop_duplicates().itertuples(index=False),
                   key=lambda r: (r.n_over_m, r.sparsity))
    conds = ["baseline", "input_at", "lat"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    width = 0.25
    xlabels = [f"n/m={c.n_over_m}\nS={c.sparsity}" for c in cells]
    x = np.arange(len(cells))
    for ax, setname in zip(axes, ["r_set1", "r_set2"]):
        for j, cond in enumerate(conds):
            means, stds = [], []
            for c in cells:
                d = con[(con.n_over_m == c.n_over_m) & (con.sparsity == c.sparsity)
                        & (con.condition == cond)][setname]
                means.append(d.mean()); stds.append(d.std())
            means = np.array(means); stds = np.array(stds)
            lo = np.minimum(stds, means * 0.999)   # keep lower whisker positive on log scale
            ax.bar(x + (j - 1) * width, means, width, yerr=[lo, stds], capsize=2, label=cond)
        ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_title(f"{setname} ({'trained dir' if setname=='r_set1' else 'held-out dirs'})")
        ax.set_ylabel("critical radius r_i (log)")
        ax.set_yscale("log")   # radii span orders of magnitude across the grid
        ax.legend()
    fig.suptitle("Claim B: robust basin width by condition (trained vs unseen directions)")
    fig.tight_layout()
    fig.savefig("results/basin_comparison.png", dpi=110)
    plt.close(fig)


def plot_concentration_vs_basin(con):
    """Claim B co-occurrence: per-cell mean PR (toward 1) vs mean held-out radius."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"baseline": "C0", "input_at": "C1", "lat": "C2"}
    con = represented(con)
    g = (con.groupby(["n_over_m", "sparsity", "condition"])
            .agg(pr=("pr", "mean"), r2=("r_set2", "mean")).reset_index())
    for cond in ["baseline", "input_at", "lat"]:
        d = g[g.condition == cond]
        ax.scatter(d.pr, d.r2, c=colors[cond], label=cond, s=60)
    ax.set_xlabel("per-concept PR (lower = more concentrated, toward 1)")
    ax.set_ylabel("held-out critical radius r_set2 (wider basin)")
    ax.set_title("Claim B: concentration vs robust-basin width (per cell)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/concentration_vs_basin.png", dpi=110)
    plt.close(fig)


def plot_proxy_scatter(con):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, p in zip(axes, ["top1_evr", "pr"]):
        d = con[np.isfinite(con[p]) & np.isfinite(con["D_groundtruth"])]
        ax.scatter(d["D_groundtruth"], d[p], s=5, alpha=0.25)
        ax.set_xlabel("ground-truth dimensionality D_i")
        ax.set_ylabel(p)
        if len(d) > 8:
            sp = spearmanr(d["D_groundtruth"], d[p]).correlation
            ax.set_title(f"{p} vs D_i (Spearman={sp:.2f})")
    fig.suptitle("Claim C: do concept proxies track ground-truth superposition?")
    fig.tight_layout()
    fig.savefig("results/proxy_vs_groundtruth.png", dpi=110)
    plt.close(fig)


# ---------- verdict ----------

def verdict(con, runs, corr, wconc):
    """Quantitative B/C verdict with across-seed spread."""
    # Claim B: per cell, does LAT widen r_set2 (held-out) vs baseline & input_at,
    # and does PR drop toward 1, co-occurring? (represented concepts only)
    con = represented(con)
    per_cell = (con.groupby(["n_over_m", "sparsity", "condition", "seed"])
                   .agg(r2=("r_set2", "mean"), r1=("r_set1", "mean"),
                        pr=("pr", "mean")).reset_index())
    cells = per_cell[["n_over_m", "sparsity"]].drop_duplicates()
    b_rows = []
    for c in cells.itertuples(index=False):
        sub = per_cell[(per_cell.n_over_m == c.n_over_m) & (per_cell.sparsity == c.sparsity)]
        piv = sub.groupby("condition").agg(r2=("r2", "mean"), r2s=("r2", "std"),
                                           r1=("r1", "mean"), pr=("pr", "mean")).to_dict("index")
        if not all(k in piv for k in ["baseline", "input_at", "lat"]):
            continue
        lat, base, inp = piv["lat"], piv["baseline"], piv["input_at"]
        b_rows.append({
            "n_over_m": c.n_over_m, "sparsity": c.sparsity,
            "r2_lat": lat["r2"], "r2_base": base["r2"], "r2_input": inp["r2"],
            "r2_lat_std": lat["r2s"],
            "lat_widens_heldout": bool(lat["r2"] > base["r2"] and lat["r2"] > inp["r2"]),
            "pr_lat": lat["pr"], "pr_base": base["pr"],
            "lat_concentrates": bool(lat["pr"] < base["pr"]),
        })
    bdf = pd.DataFrame(b_rows)
    b_widen_frac = float(bdf["lat_widens_heldout"].mean()) if len(bdf) else float("nan")
    b_cooccur_frac = float((bdf["lat_widens_heldout"] & bdf["lat_concentrates"]).mean()) if len(bdf) else float("nan")

    # Claim C: which proxies track D_i (pooled spearman) and where they diverge
    pooled = corr[corr.regime == "ALL"][["proxy", "spearman_vs_D"]].set_index("proxy")["spearman_vs_D"].to_dict()

    # Concentration at the level that can actually move (weights), not the
    # structurally-degenerate latent-cloud proxy.
    wc_conc_frac = float(wconc["lat_concentrates"].mean()) if len(wconc) else float("nan")
    wc_D_frac = float((wconc["dD_lat_minus_base"] > 0).mean()) if len(wconc) else float("nan")
    wc_I_frac = float((wconc["dI_lat_minus_base"] < 0).mean()) if len(wconc) else float("nan")

    summary = {
        "claim_B": {
            "cells_where_LAT_widens_heldout_basin": b_widen_frac,
            "cells_where_widen_AND_concentrate_cooccur_LATENTCLOUD": b_cooccur_frac,
            "_note_latentcloud": ("pr/top1_evr proxy is measured on the linear early "
                                  "latent where each feature is structurally rank-1; it "
                                  "tracks sparsity (interference), not concept cleanliness, "
                                  "so it cannot detect concentration. See weight-level below."),
            "WEIGHT_LEVEL_concentration": {
                "cells_where_LAT_raises_D": wc_D_frac,
                "cells_where_LAT_lowers_interference": wc_I_frac,
                "cells_where_both (concentrates)": wc_conc_frac,
                "per_cell": wconc.to_dict("records"),
            },
            "per_cell": b_rows,
        },
        "claim_C": {
            "pooled_spearman_proxy_vs_D": pooled,
            "note": "see results/proxy_correlations.csv for per-regime breakdown",
        },
    }
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    runs, con = load()
    corr = proxy_correlations(con)
    basin_comparison(con)
    wconc = weight_concentration(runs)
    concentration_vs_gt(con)
    plot_basin(con)
    plot_concentration_vs_basin(con)
    plot_proxy_scatter(con)
    summary = verdict(con, runs, corr, wconc)

    print("\n==== CLAIM B (concentrate + widen) ====")
    b = summary["claim_B"]
    print(f"LAT widens held-out basin vs baseline & input-AT in "
          f"{b['cells_where_LAT_widens_heldout_basin']*100:.0f}% of cells")
    print(f"[latent-cloud proxy] widen AND concentrate co-occur in "
          f"{b['cells_where_widen_AND_concentrate_cooccur_LATENTCLOUD']*100:.0f}% of cells "
          f"(proxy is structurally degenerate -- see note)")
    w = b["WEIGHT_LEVEL_concentration"]
    print(f"[weight level] LAT raises per-feature D in "
          f"{w['cells_where_LAT_raises_D']*100:.0f}% of cells; lowers interference in "
          f"{w['cells_where_LAT_lowers_interference']*100:.0f}%; concentrates (both) in "
          f"{w['cells_where_both (concentrates)']*100:.0f}%")
    print("\n==== CLAIM C (do proxies track ground truth?) ====")
    for p, s in summary["claim_C"]["pooled_spearman_proxy_vs_D"].items():
        print(f"  {p:10s} Spearman vs D_i = {s:.3f}")
    print("\nArtifacts written to results/.")


if __name__ == "__main__":
    main()
