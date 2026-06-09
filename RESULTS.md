# Results — LAT Toy-Model Study (v1)

Study implementing `SPEC.md`. Verdicts on Claim B (concentrate + widen) and
Claim C (do the proxies track ground truth?), with across-seed variance.

## Setup actually run

- **Substrate: two-layer "perturb-early / measure-late" toy** (`data.TwoLayerToyModel`).
  This is the SPEC §9 escalation, taken because the default single-bottleneck
  probe was found **analytically degenerate** (see *Substrate fork* below).
- Grid: `n/m ∈ {2,4,8}` × `S ∈ {0.8, 0.9, 0.99}` (9 cells), `m=10`.
- 3 conditions (baseline / input-AT / LAT), **5 seeds** each → **135 runs**.
- All other knobs are SPEC defaults (L2 ball, untargeted, geometric importance,
  single relative ε=0.10, 7 PGD steps, matched optimiser/steps). See `results/config.json`.
- Every Section-6 metric is logged per (cell × condition × seed): model-level in
  `results/runs.csv`, per-concept in `results/concepts.csv`.

Robust-region aggregates use **represented concepts only** (`‖W1_i‖ ≥ 0.30`);
geometric importance starves late features, and an unrepresented concept has a
degenerate (right-censored) radius. Saturation at the radius cap is logged
(`sat_set2` ≤ 0.12 everywhere, mostly 0) so the radii below are genuine, not censored.

## Claim B — concentrate + widen

**Verdict: split. The *widen* half is strongly supported and generalises to
unseen directions; the *concentrate* half is not observed; the two do NOT co-occur.**

### Widening (supported, 9/9 cells)
LAT increases the per-concept critical radius along **held-out random directions**
(`r_set2`, directions the adversary never practised) above *both* baseline and
input-AT, in **every** grid cell. Mean over represented concepts & seeds:

| n/m | S | r_set2 baseline | r_set2 input-AT | r_set2 **LAT** |
|----:|----:|----:|----:|----:|
| 2 | 0.80 | 0.552 | 0.546 | **0.862** |
| 2 | 0.90 | 0.509 | 0.518 | **0.745** |
| 2 | 0.99 | 0.425 | 0.431 | **0.623** |
| 4 | 0.80 | 0.593 | 0.721 | **0.921** |
| 4 | 0.90 | 0.560 | 0.578 | **0.858** |
| 4 | 0.99 | 0.539 | 0.526 | **1.425** |
| 8 | 0.80 | 2.090 | 4.841 | **6.035** |
| 8 | 0.90 | 2.365 | 3.170 | **5.927** |
| 8 | 0.99 | 1.569 | 2.004 | **2.207** |

The effect holds for the *worst-case / trained* direction (`r_set1`) too, but the
key point for "robust understanding" is that it holds for **set 2 (unseen)** — the
robustness generalises, it is not memorised noise. Across-seed spread is small
relative to the gap in the high-sparsity cells (e.g. n/m=2: r_set2 LAT std ≤ 0.04).
It is **large** in the two low-sparsity high-capacity cells (n/m=8, S∈{0.8,0.9}:
std ≈ 4.5), so the ordering there is directional but noisy. See
`results/basin_comparison.png`, `results/basin_comparison.csv`.

### Concentration (not observed)
LAT does **not** push concepts toward ~1D in the early latent. The cleanliness
proxy `top1_evr` is essentially unchanged across conditions (e.g. n/m=2,S=0.8:
baseline 0.306 vs LAT 0.299), and participation ratio drifts slightly *up* under
LAT, not toward 1. So in this toy:

> widening of the robust basin and concentration toward 1D are **decoupled** —
> LAT widens the neutral region **without** making the concept cleaner.

This contradicts Claim B's central prediction that the two co-occur
(`results/summary.json`: co-occurrence in 0/9 cells). See *Caveats* for the
measurement limitation on the concentration proxy.

## Claim C — do the proxies track ground-truth superposition?

**Verdict: only in the high-superposition-pressure regime. The proxies track
ground-truth dimensionality `D_i` well when capacity pressure and sparsity are
both high, and break down (≈0 correlation) otherwise. Pooling across regimes
hides this and gives a misleadingly null result.**

Spearman of the concept proxy `top1_evr` against ground-truth `D_i`, per regime:

| regime | Spearman(top1_evr, D_i) |
|---|---:|
| **pooled (ALL)** | **−0.03**  ← misleading |
| n/m=2, S=0.80 | −0.24 |
| n/m=2, S=0.90 | 0.02 |
| n/m=4, S=0.90 | −0.00 |
| n/m=4, **S=0.99** | **0.75** |
| n/m=8, **S=0.90** | **0.58** |
| n/m=8, **S=0.99** | **0.83** |

(`pr` mirrors with opposite sign, as expected.) The proxies are only trustworthy
in the regime SPEC §7 calls out as most analogous to a narrow safety concept
(high sparsity, high capacity pressure). The pooled correlation collapses to ~0
because sign and strength vary by regime — **a concrete caution for the parent
project: do not pool these proxies across regimes.** Full table per regime and
per condition: `results/proxy_correlations.csv`; scatter:
`results/proxy_vs_groundtruth.png`.

## Caveats

- **Across-seed variance.** Verdicts above are mean over 5 seeds; the held-out
  basin ordering is tight in high-sparsity cells but noisy in the two low-sparsity
  high-capacity cells (n/m=8, S∈{0.8,0.9}). Per-seed spread is in
  `results/basin_comparison.csv` and `results/summary.json`.
- **Concentration proxy.** The per-concept cloud `{h1 | i active} − mean(h1 | i inactive)`
  has its spread dominated by *co-active* features (interference), so its PR/EVR
  measures interference dimensionality more than the concept's own cleanliness.
  The "no concentration" result should be read as "the activation-cloud proxy did
  not move," not necessarily "weight-level superposition was unchanged" (the latter
  is taken as given from Gorton et al. per SPEC §2 and not re-tested here).
- **Reconstruction quality.** Two-layer FVU ≈ 0.07–0.14 (vs ~0.01–0.08 single-layer);
  the extra nonlinearity costs some fidelity but is matched across conditions.
- **Dead features.** ~40% of concepts are unrepresented at n/m≥4 under geometric
  importance and are excluded from robust-region stats (included for ground-truth
  and proxy tables where defined).

## Substrate fork — why two layers (logged per SPEC §9)

The default single-bottleneck probe was run first (preliminary). It was
**analytically degenerate**: for a linear encoder + ReLU readout, a concept's
output is `x'_i = ReLU(W_i·h + b_i)`, which only responds to the `W_i` component
of a latent perturbation, so the critical radius reduces to `r_i ≈ τ/‖W_i‖`
(verified empirically: `median(r_set1·‖W_i‖) = 0.106 ≈ τ`). It carried no
independent "neutral region" information and could not distinguish conditions.
Per SPEC §9 this is the trigger to escalate to a two-layer perturb-early /
measure-late toy, which is what these results use. The held-out direction set was
also switched from *orthogonal-to-worst-case* (which right-censored 41% of radii)
to *plain random* — still "unseen" since the adversary practised the gradient
direction, not random ones.

## Reproduce

```bash
cd lat_toy_study
python3 run.py        # full sweep -> results/runs.csv, results/concepts.csv  (~21 min, 4 cores)
python3 analyze.py    # tables, plots, results/summary.json + printed verdict
```

Artifacts: `results/proxy_correlations.csv`, `results/basin_comparison.csv`,
`results/concentration_vs_gt.csv`, `results/summary.json`, and PNGs
`basin_comparison`, `concentration_vs_basin`, `proxy_vs_groundtruth`.
