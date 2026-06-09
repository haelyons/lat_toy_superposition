# Follow-up experiments — summary

Generated from `results/followups.csv` (see `followups.py`, `CRITIQUE.md`).
Cells: low-capacity n/m=2 vs high-capacity n/m=8, S=0.9. Seeds averaged.

## 1. epsilon dose-response (LAT)

Mean over seeds; baseline shown as eps=0.00 reference.

**n/m = 2**

| eps | interference | mean_D | r_set2 (basin) |
|----:|---:|---:|---:|
| 0.00 (base) | 0.065 | 0.450 | 0.511 |
| 0.05 | 0.061 | 0.468 | 0.635 |
| 0.10 | 0.056 | 0.483 | 0.745 |
| 0.20 | 0.056 | 0.494 | 1.056 |
| 0.40 | 0.073 | 0.480 | 2.917 |

**n/m = 8**

| eps | interference | mean_D | r_set2 (basin) |
|----:|---:|---:|---:|
| 0.00 (base) | 0.331 | 0.109 | 2.507 |
| 0.05 | 0.373 | 0.111 | 4.679 |
| 0.10 | 0.382 | 0.113 | 6.333 |
| 0.20 | 0.347 | 0.112 | 7.371 |
| 0.40 | 0.212 | 0.102 | 16.052 |

## 2. targeted vs untargeted LAT (eps=0.10)

`D_target` = ground-truth dimensionality of the corrupted concept (feature 0);
higher = that concept is cleaner / more its own direction.

| n/m | condition | interference | mean_D | D_target | r_set2 | r_set2_target |
|----:|---|---:|---:|---:|---:|---:|
| 2 | baseline | 0.065 | 0.450 | 0.486 | 0.511 | 0.509 |
| 2 | lat | 0.056 | 0.483 | 0.499 | 0.745 | 0.793 |
| 2 | lat_targeted | 0.067 | 0.447 | 0.587 | 0.564 | 0.835 |
| 8 | baseline | 0.331 | 0.109 | 0.497 | 2.507 | 0.500 |
| 8 | lat | 0.382 | 0.113 | 0.600 | 6.333 | 0.966 |
| 8 | lat_targeted | 0.343 | 0.108 | 0.173 | 2.428 | 13.106 |

## Interpretation

**1. eps dose-response.** Basin widening (`r_set2`) is strictly monotone in
eps in both cells -- robustness is causally eps-driven, not a knife-edge of the
v1 eps=0.10. Concentration (n/m=2: `mean_D` up, interference down) is an
**inverted-U**: it improves up to eps in [0.10, 0.20] then degrades at eps=0.40
(over-perturbation). The v1 single eps=0.10 sat near the concentration optimum.
At n/m=8 concentration stays flat/absent across all eps -- the capacity-
dependence (Bereska) is robust to eps, not an artifact of one eps.

**2. Targeted LAT (matches Abbas).** At n/m=2, targeting feature 0 widens the
*targeted* concept's basin (`r_set2_target` up in 3/3 seeds, >= untargeted)
**without** inducing the global de-superposition that untargeted LAT produces
(global interference and `mean_D` stay ~baseline). I.e. targeting protects one
concept locally and is 'more diffuse' globally -- exactly Abbas's single-concept
concentration picture and SPEC 9's prediction. At n/m=8 the targeted result is
**degenerate and noisy**: in 2/3 seeds `r_set2_target` approaches the radius cap
(~19, right-censored) while `D_target` collapses -- the high-capacity model
flattens the targeted feature rather than cleanly protecting it. No clean
conclusion at n/m=8; reported with this caveat.
