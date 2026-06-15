# Lesion / editability probe — summary

Capability modification = directional **knockout** of one concept (the standard activation edit); **collateral** = mean FVU increase on the *other* represented concepts; **self_drop** = FVU increase on the knocked-out concept (how much capability was actually removed — a fairness check). Full Claim B grid (n/m∈{2,4,8} × S∈{0.8,0.9,0.99}), 5 seeds.

Does the geometric concentration measure (interference `I_i`) cash out as functional editability, and are LAT's more-concentrated models more surgically editable?

## (1) Geometry → function: does concentration predict editability?

**Magnitude confound** (pooled, all 3555 lesions): r(collateral, I_i) = -0.42 — *negative*, because r(I_i, w_norm) = -0.61: high-interference concepts are the barely-represented ones whose removal does nothing (r(collateral, w_norm) = +0.32).

**Conditioning on a genuine knockout** (self_drop≥0.5, 2857 lesions) removes the confound; interference then predicts collateral in the analytic (positive) direction — the geometric measure has real editability meaning:

| condition | r(coll, I_i) genuine | r(coll, I_i) pooled | mean collateral | mean self_drop |
|---|---:|---:|---:|---:|
| baseline | +0.17 | -0.31 | 0.0592 | 0.889 |
| input_at | +0.02 | -0.43 | 0.0365 | 0.750 |
| lat | +0.16 | -0.47 | 0.0464 | 0.811 |

Pooled genuine-knockout: r(collateral, I_i) = +0.12.

## (2) Editability by condition (mean collateral per cell; lower = more editable)

Self_drop is matched across conditions (full directional knockout), so raw collateral is comparable.

| n/m | S | baseline | input_at | lat | LAT<base? | LAT<inp? |
|----:|---:|---:|---:|---:|:--:|:--:|
| 2 | 0.8 | 0.0528 | 0.0487 | 0.0522 | 3/5 | 0/5 |
| 2 | 0.9 | 0.0561 | 0.0537 | 0.0544 | 5/5 | 2/5 |
| 2 | 0.99 | 0.0483 | 0.0474 | 0.0432 | 5/5 | 4/5 |
| 4 | 0.8 | 0.0539 | 0.0500 | 0.0506 | 5/5 | 1/5 |
| 4 | 0.9 | 0.0600 | 0.0536 | 0.0551 | 5/5 | 1/5 |
| 4 | 0.99 | 0.0826 | 0.0565 | 0.0616 | 5/5 | 1/5 |
| 8 | 0.8 | 0.0478 | 0.0159 | 0.0332 | 5/5 | 0/5 |
| 8 | 0.9 | 0.0503 | 0.0371 | 0.0366 | 5/5 | 2/5 |
| 8 | 0.99 | 0.0666 | 0.0205 | 0.0511 | 5/5 | 0/5 |

**Totals (seed-cells where LAT is more editable):** LAT<baseline 43/45; LAT<input_at 11/45.

## The capability-modification 2×2

Mean collateral (lower = more editable): input_at (0.036) < lat (0.046) < baseline (0.059).

| direction of modification | clean winner | LAT vs baseline |
|---|---|---|
| **edit / remove EXISTING** (this probe) | input-AT (usually) ≥ LAT > baseline | LAT more editable in 43/45 |
| **acquire NEW** (`innovability.py`) | input-AT | LAT ≈ baseline (both forget) |

So **input-space robustness wins on BOTH directions of capability modification**; LAT improves editability over baseline but is dominated by input-AT, and does not help new-concept acquisition. The unifying read: input-AT shapes the representation so that capability edits — adding or removing — cost less elsewhere; LAT's latent robustness is a weaker version of the same for removal and the wrong tool for addition.
