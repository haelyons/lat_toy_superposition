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

**Verdict (revised after the literature critique — see `CRITIQUE.md`): both
halves supported, capacity-dependently. The *widen* half is strongly supported
and generalises to unseen directions. The *concentrate* half IS observed once
measured at the level that can move (the weight geometry, not the
structurally-degenerate latent-cloud proxy); widen + concentrate co-occur in the
low/mid-capacity cells and the concentration weakens/reverses at the highest
capacity (n/m = 8), matching Bereska et al.**

> **v1 said the opposite ("concentrate not observed; decoupled"). That was a
> measurement artifact.** The concentration proxy was measured on the linear
> early latent `h1 = W1·x`, where each feature's signal is structurally rank-1, so
> the proxy tracks co-activation interference (sparsity) and cannot detect
> concentration. Disqualified — see *Concentration* below.

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

### Concentration (observed at the weight level; the latent-cloud proxy is disqualified)

**The latent-cloud proxy cannot detect concentration here.** `top1_evr`/`pr` are
computed on `h1 = W1·x`, a *linear* map, so feature *i*'s own signal is exactly
one direction (`W1_i`); an isolated feature gives `top1_evr = 1.0000`. The cloud's
apparent dimensionality is pure co-activation interference, set by sparsity
(`top1_evr` ≈ 0.27 / 0.38 / 0.90 at S = 0.80 / 0.90 / 0.99, mirroring the
co-active-feature count). It is constant across conditions because it *structurally
cannot move with training* — not because concentration is absent. v1 reported its
null as a finding; it should have been disqualified.

**Measured where it can move — the ground-truth weight geometry — LAT
concentrates** (`results/weight_concentration.csv`, paired by seed):

- **LAT raises per-feature dimensionality `D` (cleaner / less superposition) in
  9/9 cells** (5/5 seeds in 8/9). `D` is the SPEC §6 ground-truth measure.
- **LAT lowers off-diagonal interference in the low/mid-capacity cells
  (n/m ∈ {2,4})** — but **raises it at n/m = 8**.

| n/m (S=0.9) | D base→LAT | interference base→LAT | concentrates? |
|----:|---|---|:--:|
| 2 | 0.451 → **0.480** | 0.065 → **0.057** | ✅ |
| 4 | 0.226 → **0.237** | 0.140 → **0.116** | ✅ |
| 8 | 0.111 → **0.115** | 0.256 → **0.293** | ✗ (interference up) |

> So widening of the robust basin and concentration of the concept geometry
> **co-occur** in the low/mid-capacity regime — LAT carves a **wider neutral
> region around a *cleaner* (lower-dimensional) concept direction**, as Claim B
> predicted. The "high-dimensional cloud" picture in v1 was the proxy artifact.

**Capacity-dependence (Bereska et al.).** The interference reduction is present
at n/m ∈ {2,4} and reverses at n/m = 8 (highest capacity pressure) — a direct
in-house replication of Bereska et al. (2025): adversarial training does not
*universally* reduce superposition; the sign depends on capacity. This also
refutes SPEC §2's original "AT reduces superposition is settled" assumption (now
revised in SPEC §2).

**Mapping onto Abbas et al. (2025).** Abbas find LAT *concentrates* the refusal
direction (first SVD component 49%→54%). Our weight-level result is the toy
analogue and **agrees in sign**: LAT yields a *robust low-dimensional basin*, not
a high-dimensional spread. The earlier apparent contradiction with Abbas was
entirely the degenerate proxy.

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

## Follow-up experiments (SPEC §9 forks; `followups.py`, `results/followups_summary.md`)

Two focused experiments on capacity-spanning cells (n/m ∈ {2,8}, S=0.9, 3 seeds),
added in the critique to test causality and the Abbas mapping directly.

**ε dose-response (LAT).** Basin widening is **strictly monotone in ε** in both
cells (n/m=2: `r_set2` 0.51→0.64→0.75→1.06→2.92 for ε=base/0.05/0.10/0.20/0.40),
so robustness is causally ε-driven, not a knife-edge of v1's single ε. Weight-level
**concentration is an inverted-U**: `mean_D` rises and interference falls up to
ε≈0.10–0.20, then degrade at ε=0.40 (over-perturbation) — v1's ε=0.10 sat near the
concentration optimum. At n/m=8, concentration stays flat across all ε: the
capacity-dependence is robust to ε, not an ε artifact.

**Targeted LAT (matches Abbas).** Corrupting a single concept (feature 0) at n/m=2
widens that **target concept's** basin (`r_set2_target` up in 3/3 seeds, ≥ untargeted
LAT) **without** the global de-superposition untargeted LAT produces (global
interference and `mean_D` ≈ baseline). That is the Abbas single-concept picture —
targeting concentrates/protects the one concept and is "more diffuse" globally,
exactly as SPEC §9 predicted. At n/m=8 the targeted result is **degenerate and
noisy** (2/3 seeds: `r_set2_target` hits the radius cap, right-censored, while
`D_target` collapses — the model flattens the targeted feature); reported with
that caveat, no clean conclusion there.

## Evolvability probe — does the robust basin *enable* adaptation? (SPEC §10; `evolvability.py`)

This is the Wagner "robustness enables evolvability" half of the hypothesis, which
v1 deferred. Robustness (a wide, clean neutral basin) is established above; Wagner's
further claim is that the same basin makes reaching a **new** concept easier. We
pretrain with one feature held out (uniform importance), give its readout a fresh
*identical* init across conditions (so only the surrounding basin differs — this
avoids a dead-ReLU revive-or-not confound), then fine-tune the new concept in under
selection pressure identically across baseline/input-AT/LAT. Cells n/m∈{2,4,8},
S=0.9, 5 seeds. Verdict in `results/evolvability_summary.md`.

**Verdict: asymmetric, partial support.** LAT's robust basin *protects existing
concepts during adaptation* but does *not* speed acquisition of new ones — the two
faces of Wagner come apart in this toy.

- **Less forgetting (supported, 12/14 both-adapted seed-cells; 5/5 at n/m=4).**
  Adding the new concept disrupts the *old* concepts less under LAT than baseline.
  Mechanistically consistent with the established interference result: cleaner,
  more-orthogonal features → the new concept's updates collide less with existing
  ones. This is the "robustness preserved through change" facet of evolvability.
- **Slower acquisition (refutes the strong reading, 13/14 seed-cells slower), with
  one outright failure** (LAT n/m=2 seed 4 never reaches the new-concept threshold
  in budget). The wide/flat robust basin is *stiffer*: smaller gradients → slower
  descent. So LAT does **not** deliver "easier/faster access to new phenotypes."
- **New-concept readability: seed noise (3/15).** A fresh readout is *not* reliably
  more separable in the LAT basin once measured across enough seeds (at 3 seeds a
  single seed faked a strong effect — a caution logged in the summary).
- **Compositional generalisation (novel dense feature combinations): mixed.**
  input-AT is often best at high co-activation `k`; LAT sits between it and baseline.
  No clean evolvability signal.

So in this toy the analogy refines to: **robustness ↔ non-destructive integration of
new concepts, not ↔ faster reach to new ones.** A falsifiable, specific narrowing of
the Wagner import rather than a blanket confirmation. **But this single-concept probe
never raced the two robustness conditions against each other on integration quality —
the innovability follow-up below does, and it relocates the entire effect.**

## Innovability — which robustness enables *viable* innovation? (follow-up; `innovability.py`)

The evolvability probe split the Wagner thesis into "forgets less" and "slower to
acquire," reported as two traded-off signals against baseline. That is a half-result,
and it tested only LAT-vs-baseline. Wagner's evolvability is neither speed nor a single
target — it is **innovability**: the breadth of new concepts reachable *without
sacrificing existing function* (a mutation that grants a new function but breaks an
essential one is selected against). That joint criterion is also the faithful
LLM-context reading — *acquire new capabilities without catastrophic forgetting* — and
it collapses the two faces into one number.

We pretrain on a **sub-environment** (a battery of `B=6` features held inactive), then
from the same checkpoint integrate each held-out "niche" independently under selection
pressure, identical SGD across conditions (only the pretrained basin differs). Per
budget we score **raw innovability** (fraction of the battery reaching new-concept
FVU<0.20) and **clean innovability** (also keeping old-concept FVU rise <0.05 — viable).
Cells n/m∈{2,4,8}, S=0.9, 5 seeds, 270 candidate-runs. See `results/innovability.png`,
`results/innovability_summary.md`.

**Verdict: viable innovation is conferred by INPUT-space robustness, not latent
robustness — overturning the LAT-specific reading.** The effect lives entirely at the
mid-capacity cell (n/m=4) where adding a concept actually taxes existing function; at
n/m=2 (under-capacity) and n/m=8 everyone integrates cleanly.

- **At n/m=4, all three conditions *reach* the new concept (raw → 1.0 by budget), but
  only input-AT does so *cleanly*** (clean innovability **0.97** at full budget; AUC
  0.77). Baseline and LAT reach the new concept and **wreck the old ones** (clean 0.07
  and 0.17; AUC 0.03 and 0.07). Per-seed forgetting is dead consistent: input-AT ≈0.027
  (below the 0.05 viability bar at every seed), baseline ≈0.065, LAT ≈0.055.
- **LAT loses to input-AT on clean innovability in 0/15 seed-cells.** The marginal
  LAT>baseline forgetting edge from the single-concept probe survives (LAT≥baseline 4/5
  at n/m=4) but is negligible next to input-AT, which the earlier probe never raced.
- **Stiffness confirmed directly but decoupled from preservation.** Init gradient norm
  on the new concept at n/m=4: LAT **0.92** vs baseline 6.5, input-AT 5.3 — LAT's basin
  is ~7× stiffer, as the slow-acquisition result implied. Yet that stiffness buys it
  **nothing** for preservation: it forgets like baseline. So "wide/flat basin ⇒ protects
  existing function" — the mechanism the evolvability write-up inferred — is **wrong**.
- **At n/m=8 input-AT even shows slightly *negative* forgetting** (−0.009): integrating
  a new concept marginally *improves* old reconstruction.

Mechanism (intuitive, and the transferable claim): **innovation is an input-space
event** — a genuinely new feature switching on is a perturbation in feature space.
Input-AT is trained to reconstruct correctly under exactly that kind of perturbation,
so a new feature direction integrates without colliding with existing ones. LAT perturbs
the *latent* `h`, a different geometry that does not align with "a new input feature
appears," so it confers no integration protection despite a much stiffer basin. **The
type of robustness must match the type of distribution shift that innovation induces** —
and for adding concepts, that is input-space, not latent-space, robustness.

This sharpens the project's thesis rather than confirming it: robustness *does* enable
evolvability in this toy, but the link is specific to **input robustness**, and LAT —
the method under study — is on the wrong side of it for non-destructive integration.

## Lesion / editability — what does "more concentrated" *mean*? (`lesion.py`)

Claim B established that LAT *concentrates* the weight geometry (per-concept
dimensionality D↑, interference I↓), but only ever as **geometry**. Its assumed
*meaning* — that a low-interference concept can be modified without collateral damage —
was never tested. This probe closes that IOU. It performs a real capability
modification — a directional **knockout** of one concept in the latent (the standard
activation edit, `h' = h − (h·ŵ_i)ŵ_i`) — and measures **collateral**: the FVU increase
on the *other* concepts. The damage to concept *j* is mediated by `(ŵ_i·ŵ_j)`, exactly
the term summed in `I_i`, so there is an analytic expectation that collateral tracks
interference; the question is whether it cashes out in *function*. Full Claim B grid, 5
seeds, 3555 knockouts. See `results/lesion.png`, `results/lesion_summary.md`.

**(1) Concentration has functional editability meaning — but only once you control for
magnitude.** Pooled, `r(collateral, I_i) = −0.42` — the *wrong* sign, a confound:
high-interference concepts are the barely-represented junk ones (`r(I_i, ‖W_i‖)=−0.61`)
whose removal does nothing. Conditioning on a *genuine* knockout (the concept was really
removed) flips it to the predicted positive direction (`r=+0.12` pooled, +0.16–0.17 for
baseline/LAT). So geometric concentration ↔ editability holds — modestly — among
concepts the model actually relies on. The dominant determinant of "what breaks if I
delete this" is **how much the model relies on the concept** (`r(collateral,‖W_i‖)=+0.32`),
not its geometric overlap; concentration is a second-order effect on top of reliance.

**(2) Robustly-trained models are more surgically editable — but it is not specifically
a LAT effect.** Mean collateral: **input-AT 0.036 < LAT 0.046 < baseline 0.059** (self-
drop matched, so this is comparable). LAT beats baseline in **43/45** seed-cells, but
loses to input-AT in **34/45**. Strikingly, input-AT has the *highest* mean interference
(0.18 vs LAT 0.14, baseline 0.13) yet the *lowest* collateral — so its editability is
**not** explained by the interference metric at all (its scatter slope is ~flat). Input-AT
achieves clean edits by some other route (plausibly the input-robust map keeps each
concept's *contribution* recoverable even when directions overlap).

**The capability-modification 2×2.** Combined with the innovability result:

| direction of modification | clean winner | LAT vs baseline |
|---|---|---|
| **remove/edit an existing concept** (lesion) | input-AT (usually) ≥ LAT > baseline | LAT more editable 43/45 |
| **acquire a new concept** (innovability) | input-AT | LAT ≈ baseline (both forget) |

**Input-space robustness wins on *both* directions of capability modification; LAT is a
weaker cousin for removal and the wrong tool for addition.** So the honest answer to
"what does more-concentrated mean": at the per-concept level it means *modestly* more
editable (geometry → function, confirmed but second-order to reliance); at the model
level, editability is a property conferred by adversarial training broadly — and best by
the input-space variety — rather than by the concentration metric per se. The clean
"LAT specialises in editing existing capabilities" hypothesis is **not** supported.

## Selectivity — does general LAT concentrate *all* concepts, or selectively? (`selectivity.py`)

Abbas et al. apply *general* (untargeted) LAT and report that *one* concept (refusal)
concentrates. If LAT concentrated everything uniformly that would be unremarkable; the
publishable question is whether general LAT is general in its *perturbation* but
**selective in its *effect***. We test this directly by regressing the LAT-induced change
in each concept's concentration against the concept's **importance** (geometric decay,
varies within a cell) and **frequency** (1−sparsity, across cells). Untargeted `lat` vs
baseline, full grid, 3 seeds. See `results/selectivity_summary.md`,
`results/selectivity.png`, and the transformer control `results/tx_selectivity.csv`.

**Verdict: general LAT is NOT uniform — it is selective, and the selection axis is set by
the fine-tuning distribution/objective.** This is the correction to any "LAT reshapes
every concept equally" reading: refusal concentrated because it is *salient to the safety
objective*, not because LAT touches all concepts the same.

- **Toy (importance = an explicit loss weight): LAT selectively strengthens/cleans the
  IMPORTANT concepts.** `r(w_norm gain, importance) = +0.46`; `r(D gain, importance) =
  +0.23` (concentration regime). input-AT shows *no* such grading (w_norm r = −0.22), so
  this is LAT-specific, not a generic "important concepts grow" artifact.
- **Transformer control (ops differ only in FREQUENCY, equal per-instance loss):** the
  sign **flips** — `r(frequency, Δembedding-norm under LAT) = −0.67` (per-seed
  −0.87/−0.41/−0.96). With equal loss weight, LAT strengthens the **rare/most-vulnerable**
  ops most (robustness-equalisation). So the *direction* of selectivity depends on what
  makes a concept matter to the loss (loss-weight vs frequency) — exactly the
  "depends on the fine-tuning data" caveat, made concrete.
- **The grading lives in the WEIGHT/reliance geometry, not the activation effective-dim.**
  The Abbas-style activation "collapse to 1D" (late-site `top1_evr`/`pr`) is ~flat vs
  importance in the toy and noisy in the transformer; what moves selectively is how
  strongly/cleanly a concept is *written into the weights*. **Audit LAT by reliance, not
  SVD spectra alone.**

This also reconnects to the lesion result: the *reliance* axis that dominated editability
is the same axis along which LAT selectively reshapes concepts.

## Realer substrate — does the story survive attention + depth? (`real/`)

Everything above lives in an Elhage-style autoencoder. The natural objection is that the
verdicts are artifacts of a linear bottleneck. So we ported the two capability-
modification probes to a **2-layer transformer** on a **multi-skill modular-arithmetic**
task (`a op b =`, 6 operations; each operation is a separable "capability"). LAT and
input-AT are applied the way they are used in practice — as a **fine-tune** on top of a
shared clean pretrain (from-scratch AT at ε=0.1 prevented the model from learning the
exact-answer task at all, itself a finding), so all three conditions branch from the same
checkpoint and only the objective differs. 3 seeds. See `results/tx.png`,
`results/tx_summary.md`, `real/`.

**Robustness sanity (the regimes genuinely diverged).** Under PGD, baseline is fragile at
both sites (adv acc ~0.2–0.3); **input-AT is robust at the input/embedding site (0.95)**
and **LAT at the latent/residual site (1.00)** — each robust exactly where it perturbs.

**The toy's central result replicates — input-space robustness wins both directions of
capability modification, cleanly.**

| direction | metric (lower = better) | baseline | input-AT | LAT | paired |
|---|---|---:|---:|---:|---|
| **remove** (lesion) | collateral of op knockout | 0.045 | **0.007** | 0.035 | input-AT>base 3/3; LAT>base 3/3; LAT>input-AT 0/3 |
| **add** (innovability) | old-skill forgetting | 0.479 | **0.432** | 0.440 | input-AT<base 7/9; LAT<base 7/9; LAT<input-AT 3/9 |

- **Editability is the strong effect**: input-AT is **~6× more surgically editable** than
  baseline (knocking out one skill barely touches the others), LAT roughly halfway —
  exactly the toy's ordering, and exactly the "LAT is a weaker cousin" pattern (LAT never
  beats input-AT, 0/3).
- **The geometry→function link replicates, more cleanly than in the toy**: the
  interference of a capability's *input (embedding) direction* positively predicts the
  collateral of removing it (`r = +0.30…+0.65` across conditions, vs the toy's +0.12).
  This reinforces the input-space theme — a capability's editability is set by how its
  *input representation* overlaps with others.
- **Innovability is directionally consistent but weaker**: with no rehearsal all regimes
  forget a lot (~45%), but both robust regimes forget less than baseline (7/9 each), with
  input-AT marginally ahead of LAT.

**Bottom line: the autoencoder verdicts were not artifacts of the linear bottleneck.**
With attention and depth, input-space adversarial training is again the consistent winner
for capability modification — most editable, least forgetting — and LAT again a weaker
version that still beats baseline. The "input robustness, not latent robustness" thesis
holds across both substrates.

*Toward the GPU substrate.* This custom-trained transformer is a stand-in: it lets us
keep the three-regime comparison (which needs us to train the models) on CPU until the
environment is wired to a GPU cluster over SSH. The two probes are written to be
substrate-agnostic — a lesion is "project a concept's direction out and measure collateral
on the rest"; innovability is "adapt a held-out capability and measure forgetting" — so on
the GPU rung they re-point at a **pretrained LLM with SAE features as the concepts** with
no change to the experimental logic. The open question that only the GPU rung can settle:
does the input-AT advantage survive when the three regimes are *full pretraining runs of a
real LLM*, rather than a fine-tune on a shared base? (`real/model.py` is plain PyTorch and
moves to CUDA unchanged; only a `.to(device)` plumbing pass is needed.)

## Caveats

- **Across-seed variance.** Verdicts above are mean over 5 seeds; the held-out
  basin ordering is tight in high-sparsity cells but noisy in the two low-sparsity
  high-capacity cells (n/m=8, S∈{0.8,0.9}). Per-seed spread is in
  `results/basin_comparison.csv` and `results/summary.json`.
- **Concentration proxy (now disqualified, see Concentration above).** The
  per-concept cloud `{h1 | i active} − mean(h1 | i inactive)` is measured on a
  *linear* encoder, so its spread is pure co-activation interference and it cannot
  move with training. It is retained only as a Claim-C calibration result (a proxy
  that tracks sparsity, not superposition). Weight-level superposition was **not**
  taken as given — it is tested directly via `D`/interference and does move.
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
