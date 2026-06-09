# Evolvability probe — summary

Wagner 'robustness enables evolvability' test (SPEC §10). Mean over seeds; paired LAT-vs-baseline by seed. `fvu_new_start` = zero-shot readability of a fresh new-concept direction (lower = more accessible niche in the basin); `forgetting` = old-concept FVU increase during adaptation (lower = cleaner integration); `steps_to_new_thresh` = SGD steps for new-concept FVU<0.20.

## Adaptation to a new concept

| n/m | condition | fvu_new_start | steps→thresh | forgetting | fvu_new_end |
|----:|---|---:|---:|---:|---:|
| 2 | baseline | 1.65 | 300 | +0.0482 | 0.000 |
| 2 | input_at | 1.83 | 330 | +0.0361 | 0.001 |
| 2 | lat | 1.48 | 300 | +0.0302 | 0.325 |
| 4 | baseline | 1.88 | 360 | +0.0701 | 0.004 |
| 4 | input_at | 1.59 | 330 | +0.0201 | 0.004 |
| 4 | lat | 1.41 | 480 | +0.0519 | 0.004 |
| 8 | baseline | 1.47 | 330 | +0.0285 | 0.016 |
| 8 | input_at | 1.36 | 300 | -0.0136 | 0.004 |
| 8 | lat | 1.42 | 420 | +0.0213 | 0.007 |

## LAT vs baseline (paired by seed; per-seed win counts, not cell means)

Means can be dominated by a single seed, so we count seed-cells where LAT beats baseline. Speed & forgetting are counted only over seeds where BOTH conditions actually adapted (a non-adapter has sentinel steps=-1 and ~0 forgetting, which would otherwise score as spurious wins).

Non-adapters (never reached new-concept FVU<0.2 in budget): {'lat': 1}.

| n/m | readability better | forgetting lower (both-adapted) | adapts faster (both-adapted) |
|----:|---:|---:|---:|
| 2 | 1/5 | 3/4 | 0/4 |
| 4 | 1/5 | 5/5 | 1/5 |
| 8 | 1/5 | 4/5 | 0/5 |

**Totals:** readability better in 3/15 seed-cells; lower forgetting in 12/14; faster adaptation in 1/14 (both-adapted seed-cells).

## Compositionality (FVU on novel dense combinations, k features co-active)

Lower = better reconstruction of unseen combinations.

**n/m=2** — k=1 | k=2 | k=4 | k=8 | k=16
| condition | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|
| baseline | 0.003 | 0.044 | 0.128 | 0.344 | 1.319 |
| input_at | 0.000 | 0.029 | 0.097 | 0.277 | 1.086 |
| lat | 0.006 | 0.044 | 0.118 | 0.311 | 1.175 |

**n/m=4** — k=1 | k=2 | k=4 | k=8 | k=16
| condition | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|
| baseline | 0.150 | 0.207 | 0.351 | 0.603 | 0.914 |
| input_at | 0.324 | 0.353 | 0.428 | 0.569 | 0.809 |
| lat | 0.211 | 0.261 | 0.380 | 0.580 | 0.854 |

**n/m=8** — k=1 | k=2 | k=4 | k=8 | k=16
| condition | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|
| baseline | 0.763 | 0.659 | 0.643 | 0.702 | 0.835 |
| input_at | 0.910 | 0.796 | 0.762 | 0.763 | 0.812 |
| lat | 0.804 | 0.692 | 0.672 | 0.713 | 0.820 |
