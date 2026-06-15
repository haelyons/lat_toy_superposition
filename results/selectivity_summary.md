# Selectivity ablation — is LAT's concentration uniform or importance-graded?

Untargeted LAT (the `lat` condition; same as Abbas et al.'s general LAT). For each concentration measure, **gain** = how much MORE concentrated a concept is under the condition vs baseline (signed so + = more concentrated). If the effect is uniform, gain is unrelated to importance; if selective, gain rises with importance.

Importance = geometric decay within a cell; frequency = 1−sparsity across cells. Concentration regime = n/m∈{2,4} (where LAT concentrates); n/m=8 saturates.

## lat vs baseline

| measure | r(gain, importance) ALL | r(gain, imp) n/m∈{2,4} | r(gain, freq) | mean gain |
|---|---:|---:|---:|---:|
| D_i | +0.17 | +0.23 | +0.07 | +0.0271 |
| I_i | -0.09 | +0.03 | +0.19 | +0.0179 |
| w_norm | +0.46 | +0.45 | +0.11 | +0.3945 |
| late_pr | -0.03 | -0.06 | +0.08 | -0.1333 |
| late_top1evr | +0.03 | +0.07 | +0.15 | -0.0040 |

## input_at vs baseline

| measure | r(gain, importance) ALL | r(gain, imp) n/m∈{2,4} | r(gain, freq) | mean gain |
|---|---:|---:|---:|---:|
| D_i | +0.04 | +0.00 | -0.18 | +0.0220 |
| I_i | +0.00 | +0.14 | +0.13 | -0.0041 |
| w_norm | -0.22 | -0.24 | +0.20 | -0.1767 |
| late_pr | +0.15 | +0.24 | +0.28 | +0.2926 |
| late_top1evr | +0.22 | +0.26 | +0.16 | +0.0258 |

## Read

- **Reliance/magnitude (w_norm)**: r(gain, importance) = +0.46 — LAT strengthens important concepts far more than marginal ones.
- **Weight cleanliness (D_i, concentration regime)**: r = +0.23.
- A positive importance correlation = **general LAT is selective**: it concentrates the concepts the loss relies on, not all concepts uniformly. This is the mechanism by which Abbas et al.'s *general* LAT produced a *refusal-specific*-looking effect. See the transformer control (`results/tx_selectivity.csv`) for the same test with skewed op frequency.

## Transformer control (skewed op-frequency; `real/selectivity.py`)

Same question with attention + depth. Ops differ only in FREQUENCY (equal per-instance loss weight, unlike the toy's importance weighting). We correlate each op's LAT-induced embedding-norm change (the w_norm analogue) with its frequency.

- r(frequency, Δembedding-norm under LAT) = **-0.67** (per-seed [-0.87, -0.41, -0.96]).
- r(frequency, activation-cloud concentration gain) = +0.38 (noisy).

**The sign is NEGATIVE — opposite the toy.** With equal per-instance loss, LAT strengthens the RARE/most-vulnerable ops most (robustness-equalisation), whereas the toy's loss-importance weighting made LAT strengthen the IMPORTANT concepts. Both refute uniformity; the selection *axis* (and its sign) is set by what makes a concept salient to the loss — i.e. by the fine-tuning distribution/objective.

## Conclusion (for the revised write-up)

1. **General (untargeted) LAT does NOT concentrate all concepts uniformly** — its effect is selective, allocated by how the loss/adversary engages each concept. This is the correction to any reading of Abbas et al. that assumed a uniform effect: refusal concentrated because it is salient to the safety objective, not because LAT touches every concept equally.
2. **The selection axis is set by the fine-tuning distribution/objective.** Loss-importance weighting → LAT concentrates/strengthens the IMPORTANT concepts (toy: w_norm r=+0.46, D r=+0.23). Pure frequency skew at equal loss weight → LAT strengthens the RARE/vulnerable concepts (transformer: embnorm-vs-freq r=-0.67).
3. **The grading is on the WEIGHT/reliance geometry, not the activation effective-dimension.** The Abbas-style activation 'collapse to 1D' (late_pr/top1evr) is ~flat vs importance in the toy and noisy in the transformer; what moves selectively is how strongly/cleanly a concept is *written into the weights*. Measure reliance, not just SVD spectra, when auditing LAT.
