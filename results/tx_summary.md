# Realer substrate — tiny transformer, three regimes — summary

Multi-skill modular arithmetic (`a op b =`, 6 operations), 2-layer transformer. Shared clean pretrain → condition-specific fine-tune (LAT/input-AT as the fine-tuning methods they are in practice), eps=0.05. Then the two capability-modification probes. Seeds = [0, 1, 2].

## Robustness sanity (did the regimes diverge?)

Accuracy under PGD at the input-embedding site and the residual (LAT) site. Higher = more robust there.

| condition | clean | adv (emb/input site) | adv (resid/latent site) |
|---|---:|---:|---:|
| baseline | 1.00 | 0.21 | 0.31 |
| input_at | 1.00 | 0.95 | 0.87 |
| lat | 1.00 | 0.49 | 1.00 |

## Lesion — editability (collateral of an op knockout; lower = more surgical)

Capability = an operation, knocked out at its op-token embedding. Self_drop is the accuracy lost on the knocked-out op (the edit landed); collateral is the mean accuracy lost on the OTHER ops.

| condition | mean collateral | mean self_drop | r(collateral, I_i) genuine |
|---|---:|---:|---:|
| baseline | 0.0446 | 0.75 | +0.48 |
| input_at | 0.0069 | 0.72 | +0.30 |
| lat | 0.0350 | 0.74 | +0.65 |

Paired by seed: input-AT more editable than baseline 3/3; LAT more editable than baseline 3/3; LAT more editable than input-AT 0/3.

## Innovability — bolt on a new skill with NO rehearsal

Adapt a held-out op (clean SGD, new op only) into each regime's basin. `new_acc_end` = acquired the new skill; `forgetting` = old-skill accuracy lost (lower = the basin integrates the new skill more cleanly).

| condition | new_acc_end | old_acc_end | forgetting |
|---|---:|---:|---:|
| baseline | 1.00 | 0.52 | +0.479 |
| input_at | 1.00 | 0.57 | +0.432 |
| lat | 1.00 | 0.56 | +0.440 |

Paired by (held-out op, seed), n=9: input-AT forgets less than baseline 7/9; LAT forgets less than baseline 7/9; LAT forgets less than input-AT 3/9.

## Verdict (does the toy story survive attention + depth?)

- Editability (lesion collateral, lower better): input_at(0.007) < lat(0.035) < baseline(0.045).
- Innovability (forgetting, lower better): input_at(+0.432) < lat(+0.440) < baseline(+0.479).

Compare to the toy: input-AT won BOTH directions of capability modification there. The table above says whether that holds with attention + depth.
