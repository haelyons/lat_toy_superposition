# Critique & Revision — LAT Toy-Model Study

A critical review of the v1 study (`SPEC.md`, `RESULTS.md`, code) against the
literature (`LITERATURE_REVIEW.md`, centred on Abbas et al. 2025), with the
highest-value fixes implemented. All v1 numbers were spot-checked and reproduce
**bit-identically** on re-train (deterministic seeds), so the issues below are
about *what was measured and claimed*, not computational error.

## Headline

The v1 verdict — *"LAT widens the robust basin but does **not** concentrate
concepts; the two are decoupled"* — is **half wrong**, and the wrong half flips
the study's relationship to the literature. Concentration was measured with a
proxy that **cannot** detect it by construction. At the level that can move (the
weight geometry), **LAT does concentrate**, and it does so *capacity-dependently*
— which simultaneously reconciles Abbas et al. (concentration) and Bereska et al.
(capacity-dependence). After the fix, widen + concentrate **co-occur** in the
low/mid-capacity cells, exactly as Claim B predicted.

Separately, the Wagner *evolvability* payoff was tested (`evolvability.py`,
`innovability.py`) and yields a sharp, surprising result: **viable innovation —
acquiring a new concept without catastrophic forgetting of old ones — is conferred by
input-space adversarial training, not by LAT.** LAT's basin is markedly stiffer yet
forgets like baseline during integration; only input-AT innovates cleanly, and only at
the capacity where it matters. The robustness↔evolvability link is real but located in
input robustness, with LAT on the wrong side of it.

## Prioritised findings

### P0 — The concentration proxy is structurally degenerate (corrected)

`metrics.concept_proxies` measures `top1_evr` / `pr` of the per-concept cloud
`{h1 | i active} − mean(h1 | i inactive)`, on the **early latent** `h1 = W1·x`.
Because that encoder is **linear**, feature *i*'s own contribution to `h1` is
exactly `x_i · W1_i` — a single direction. The cloud's apparent dimensionality is
therefore *entirely* co-activation interference from other features, which is set
by **sparsity**, not by how cleanly concept *i* is encoded. Demonstrated:

| measured on baseline model | S=0.80 | S=0.90 | S=0.99 |
|---|---:|---:|---:|
| cloud `top1_evr` (feature 0) | 0.27 | 0.38 | 0.90 |
| cloud `pr` | 7.1 | 5.1 | 1.2 |
| expected #co-active features `≈ n(1−S)` | 4.0 | 2.0 | 0.2 |

The proxy *is* the co-activation count. Isolating a single feature (no
co-activation) gives `top1_evr = 1.0000` — structurally rank-1. So the proxy can
**never** move with training; "concentration not observed" was a near-tautology,
not evidence. The v1 RESULTS caveat noted the proxy is "interference-dominated"
but still reported the null as a finding contradicting Claim B; it should have
been disqualified.

**Fix:** measure concentration where it can move — the ground-truth weight
geometry, which the pipeline already logs (`mean_D`, off-diagonal `interference`)
but never compared across conditions. New `analyze.weight_concentration` does the
paired-by-seed comparison. Result (`results/weight_concentration.csv`):

- **LAT raises per-feature dimensionality `D` (cleaner / less superposition) in
  9/9 cells**, 5/5 seeds in 8/9 cells.
- **LAT lowers off-diagonal interference in the low/mid-capacity cells
  (n/m ∈ {2,4})** but **raises it at n/m = 8** (highest capacity pressure).

So LAT *does* concentrate the concept geometry, and **widen + concentrate
co-occur** (both true) in the low/mid-capacity regime — the opposite of the v1
verdict, and what Claim B predicted.

### P1 — This reconciles the central literature tension (Abbas vs our v1)

The review framed a tension: Abbas et al. find LAT *concentrates* refusal (SVD
explained variance up), denoising/purification theory points to *lower* effective
dimension, yet our v1 found *no* concentration. The tension was an artifact of P0.
With the weight-level metric:

- **Agrees with Abbas et al. / Allen-Zhu & Li / denoising-AE theory:** LAT makes
  the concept geometry cleaner (higher `D`, lower interference) — i.e. a **robust
  *low-dimensional* basin**, not a high-dimensional cloud. The right reframing of
  Claim B is "robust low-D basin," exactly as the literature review's Stage-2
  recommendation anticipated. The basin-*widening* result (v1, holds) and the
  concentration result are then two faces of the same thing: LAT carves a wider
  neutral region around a *cleaner* concept direction.
- **Resolves the "cloud vs basin" question:** there is no real high-dimensional
  cloud — that appearance was the interference proxy. Drop the "high-dimensional
  cloud" framing entirely.

### P1 — Engage Bereska et al. (capacity-dependence) and fix SPEC §2

The interference reduction is **capacity-dependent**: present at n/m ∈ {2,4},
absent/reversed at n/m = 8 (interference rises under LAT in 4/5 seeds there). This
is a direct, in-house instance of Bereska et al. (2025): "adversarial training
does not universally reduce superposition; its effect depends on task complexity
relative to network capacity." **SPEC §2's assumption that "AT reduces
superposition, taken as settled (Gorton et al.)" is contradicted by our own
data** and by the literature, and is now flagged as contested rather than assumed.
The v1 RESULTS even leaned on this assumption to excuse the null ("weight-level
superposition is taken as given … not re-tested") — but the weight metrics *were*
logged and *do* move, so we test it directly instead of assuming it.

### P2 — ε-sweep (dose-response; SPEC §9 fork) — NEW DATA

v1 used a single ε=0.10, so widening/concentration could be a knife-edge artifact.
`followups.py` sweeps ε ∈ {0.05, 0.10, 0.20, 0.40} on a low- vs high-capacity cell
(n/m ∈ {2,8}, S=0.9, 3 seeds). **Outcome:** basin widening is **strictly monotone
in ε** in both cells (causal, not a knife-edge); weight-level concentration is an
**inverted-U** (improves to ε≈0.10–0.20, degrades at ε=0.40 over-perturbation), so
v1's ε=0.10 sat near the concentration optimum. At n/m=8 concentration stays flat
across all ε — the capacity-dependence is not an ε artifact. Full tables:
`results/followups_summary.md`.

### P2 — Targeted LAT (matches Abbas; SPEC §9 fork) — NEW DATA

Abbas's result is on a *single* concept. v1 only ran untargeted LAT. New
`lat_targeted` condition (`train.py`): the adversary corrupts **one** feature
(the most important, j=0); the defender still optimises full reconstruction. We
track the **target feature's own** geometry (`D_target`), not just the global
average. **Outcome (n/m=2):** targeting widens the *targeted* concept's basin
(`r_set2_target` up in 3/3 seeds, ≥ untargeted LAT) **without** the global
de-superposition untargeted LAT produces (global interference/`mean_D` ≈ baseline)
— i.e. it protects one concept locally and is "more diffuse" globally, exactly
Abbas's single-concept picture and SPEC §9's prediction. **(n/m=8):** degenerate
and noisy (2/3 seeds right-censor `r_set2_target` at the radius cap while `D_target`
collapses — the model flattens the targeted feature); no clean conclusion there.
See `results/followups_summary.md`.

### P1 — Evolvability probe (the Wagner payoff; SPEC §10) — NEW DATA

The robustness half of the hypothesis was done; the *evolvability* half — the whole
point of the Wagner import — was untested. `evolvability.py` adds it: pretrain with
one concept held out, give it a fresh identical readout, then fine-tune it in
identically across conditions and measure adaptation speed, final quality, and
forgetting of old concepts (n/m∈{2,4,8}, S=0.9, 5 seeds). Two design bugs were found
and fixed en route — a dead-ReLU revive-or-not confound (fixed by the fresh readout)
and a 3-seed result that was entirely seed-0-driven (fixed by 5 seeds + per-seed
win-counts, not cell means). **Outcome: asymmetric, partial support.** LAT
*reduces forgetting* of existing concepts during adaptation (12/14 seed-cells) but is
*slower* to acquire the new concept (13/14) and fails once outright — appearing to
protect what exists while being stiffer, not more plastic. Readability (3/15) and
compositionality are not LAT-favouring. See `results/evolvability_summary.md`.

### P1 — Innovability follow-up relocates the evolvability effect — NEW DATA

The evolvability probe raced only LAT-vs-baseline and split the result into "forgets
less" + "slower." `innovability.py` tests the evolutionarily honest joint criterion —
**viable innovation**: across a battery of 6 held-out niches, what fraction can be
acquired (new-concept FVU<0.20) *while preserving existing function* (old-concept FVU
rise<0.05)? This collapses the two faces into one number and, crucially, races all
three conditions. **Outcome: viable innovation is conferred by INPUT-space robustness,
not latent robustness — the LAT-specific reading is overturned.** At the only
discriminating cell (n/m=4, where innovating taxes capacity), all conditions *reach* the
new concept but only input-AT integrates it *cleanly* (clean innovability 0.97 vs
baseline 0.07, LAT 0.17; per-seed forgetting input-AT ≈0.027 vs baseline 0.065, LAT
0.055). **LAT loses to input-AT on clean innovability in 0/15 seed-cells.** The
gradient-norm probe confirms LAT's basin is ~7× stiffer (0.92 vs 5–6.5) yet that
stiffness buys *nothing* for preservation — so the "wide basin protects existing
function" mechanism inferred above is **wrong**. Mechanism: adding a concept is an
input-space perturbation, which input-AT is trained for; LAT perturbs the latent, a
mismatched geometry. **The kind of robustness must match the kind of shift innovation
induces.** See `results/innovability.png`, `results/innovability_summary.md`.

### P1 — Lesion probe: what "concentration" *means* for capability modification — NEW DATA

Claim B measured concentration as geometry (interference I↓) but never tested its
assumed *meaning*: that a low-interference concept is editable without collateral damage.
`lesion.py` performs a real capability edit — directional **knockout** of a concept — and
measures collateral FVU on the others (full grid, 5 seeds, 3555 knockouts). **Outcome,
two parts.** (1) The geometry→function link is real but **second-order and confounded**:
pooled `r(collateral, I_i)=−0.42` has the wrong sign purely because high-I concepts are
weakly-represented junk; conditioning on a genuine knockout flips it to the predicted
`+0.12` (+0.16 baseline/LAT). What actually governs "what breaks if I delete this" is
*reliance* (`r(collateral,‖W_i‖)=+0.32`), not concentration. (2) Robustly-trained models
*are* more editable (mean collateral input-AT 0.036 < LAT 0.046 < baseline 0.059; LAT
beats baseline 43/45) — but it is **not a LAT-specific or even an interference-driven
effect**: input-AT has the *highest* interference yet the *lowest* collateral. Combined
with innovability this yields a capability-modification 2×2 in which **input-space
robustness wins both directions** (remove *and* add), with LAT a weaker cousin on removal
and the wrong tool for addition. So "more concentrated" cashes out as only a modest,
second-order editability gain — the clean "concentration ⇒ surgical editability ⇒ LAT's
advantage" story is **not** supported. See `results/lesion.png`,
`results/lesion_summary.md`.

### P1 — Realer substrate (transformer) replicates the input-AT advantage — NEW DATA

The obvious objection to all of the above: it is a linear autoencoder. `real/` ports both
capability-modification probes to a **2-layer transformer** on multi-skill modular
arithmetic (6 operations as separable capabilities), with LAT/input-AT applied as
fine-tunes on a shared clean pretrain (their real-world usage; from-scratch AT at ε=0.1
could not even learn the task). Robustness sanity confirms the regimes diverged (input-AT
robust at the embedding site 0.95, LAT at the residual site 1.00, baseline neither).
**The toy's central result replicates cleanly:** input-AT is the most surgically editable
(knockout collateral 0.007 vs baseline 0.045 vs LAT 0.035; input-AT>baseline 3/3, LAT
never beats input-AT) and forgets least when a new skill is bolted on with no rehearsal
(input-AT<baseline 7/9). The geometry→function link is *stronger* here than in the toy —
a capability's input-embedding interference predicts its knockout collateral at
`r=+0.30…+0.65`. So the "input robustness, not latent robustness, drives capability
modification; LAT is a weaker cousin" thesis is **not** an artifact of the linear
bottleneck. (CPU stand-in for the future GPU-cluster rung, where the same probes re-point
at a pretrained LLM + SAE features.) See `results/tx.png`, `results/tx_summary.md`.

## What was NOT changed (and why)

- **Main 135-run sweep not re-run.** It reproduces exactly and the fix is an
  analysis/interpretation change, not a data change. Re-running would waste ~20
  min and change nothing.
- **Latent-cloud proxy left in the pipeline** (not deleted) but demoted in the
  verdict with an explicit "structurally degenerate" note — it is still a fair
  *calibration* result for Claim C (it shows a proxy that tracks sparsity, not
  superposition, which is itself a useful caution for the parent project).
- **No SAE-superposition rewrite.** The existing SAE monosemanticity barely moves
  across conditions and is noisy; a Bereska-style SAE superposition metric would
  be a larger build with low marginal value given the weight-level metric already
  delivers the capacity-dependence result cleanly. Deferred.

## Success criteria (defined before implementing)

1. Weight-level concentration computed paired-by-seed, per cell, written to CSV,
   and surfaced in the verdict. ✅ (`results/weight_concentration.csv`, summary.json)
2. Verdict text corrected: concentration **is** observed at the weight level; the
   latent-cloud null is disqualified. ✅
3. ε-sweep and targeted-LAT produce new data on ≥2 capacity-spanning cells, ≥3
   seeds, with a written dose-response / targeting outcome. ✅ (`followups.py`)
4. RESULTS.md and SPEC §2 updated to reflect the corrected story and the
   capacity-dependence caveat. ✅
