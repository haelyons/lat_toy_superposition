# Innovability probe — summary

Wagner's evolvability = **innovability**: breadth of new concepts reachable *without sacrificing existing function*. `evolvability.py` found LAT forgets less but is stiffer; here we test the joint, evolutionarily honest criterion (innovate while remaining viable) across a battery of 6 held-out niches, 5 seeds, cells n/m∈{2,4,8}, S=0.9.

**reached** = new-concept FVU<0.2. **clean** = reached AND old-concept FVU rise<0.05 (function preserved). Budget = 1500 SGD steps.

## Stiffness (direct): mean ‖∇‖ on the new concept at adaptation step 0

Smaller = stiffer basin (slower descent), the mechanism behind slower acquisition.

| n/m | baseline | input_at | lat |
|----:|---:|---:|---:|
| 2 | 3.620 | 3.489 | 2.149 |
| 4 | 6.523 | 5.319 | 0.921 |
| 8 | 3.066 | 2.447 | 1.028 |

## Innovability vs budget (fraction of battery)

raw = reached; clean = reached & viable. Shown at an early budget (150 steps) and full budget (1500 steps).

| n/m | condition | raw@150 | raw@1500 | clean@150 | clean@1500 |
|----:|---|---:|---:|---:|---:|
| 2 | baseline | 0.77 | 1.00 | 0.77 | 1.00 |
| 2 | input_at | 0.77 | 1.00 | 0.77 | 1.00 |
| 2 | lat | 0.53 | 0.93 | 0.50 | 0.93 |
| 4 | baseline | 0.00 | 1.00 | 0.00 | 0.07 |
| 4 | input_at | 0.07 | 1.00 | 0.07 | 0.97 |
| 4 | lat | 0.00 | 1.00 | 0.00 | 0.17 |
| 8 | baseline | 0.00 | 1.00 | 0.00 | 1.00 |
| 8 | input_at | 0.27 | 1.00 | 0.27 | 1.00 |
| 8 | lat | 0.00 | 1.00 | 0.00 | 1.00 |

## Budget-integrated clean innovability (area under clean curve, 0–1)

One scalar per condition: higher = more viable novelty accessible across budgets.

| n/m | baseline | input_at | lat |
|----:|---:|---:|---:|
| 2 | 0.881 | 0.910 | 0.806 |
| 4 | 0.032 | 0.771 | 0.074 |
| 8 | 0.788 | 0.861 | 0.772 |

## Paired win-counts (per seed, battery-averaged): clean innovability

Counts seeds where LAT ≥ comparator on budget-integrated clean innovability (ties to LAT broken as wins only if strictly ≥; reported as wins/seeds).

| n/m | LAT ≥ baseline | LAT ≥ input_at |
|----:|---:|---:|
| 2 | 2/5 | 0/5 |
| 4 | 4/5 | 0/5 |
| 8 | 1/5 | 0/5 |

**Totals:** LAT ≥ baseline in 7/15 seed-cells; LAT ≥ input_at in 0/15.
