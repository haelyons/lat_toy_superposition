# SPEC — LAT Toy-Model Study (v1)

### 1. Goal (plain language)

We want to know whether training a model against noise in its own internal activations (“latent adversarial training,” LAT) changes *how it represents concepts* — specifically, whether it makes each concept’s representation more robust.

We cannot measure this cleanly in a real language model, because you cannot directly observe how many concepts (features) a real model packs into its activation space. So we use a toy model where we *generate* the features ourselves and therefore know the ground truth. This lets us check both what LAT does and whether the indirect measurements (“proxies”) used on real models are trustworthy.

### 2. The theory, simply

Picture a concept as living somewhere in the model’s internal activation space. Two pictures of it:

- **Fragile point** — the concept is a single precise location; nudge the activations slightly and the concept is lost or corrupted.
- **Robust region** — the concept occupies a whole connected region; you can move around inside it and the concept’s identity is preserved.

Our hypothesis borrows an idea from evolutionary biology: robustness as a large “neutral region” a system can move around within without changing its function. We think LAT pushes concepts from the fragile-point picture toward the robust-region picture. A genuinely *robust understanding* of a concept would mean (a) the region is wide, and (b) the robustness is **general** — the concept survives perturbations in directions training never specifically practised, not only the ones it was trained against. If a concept survives only the exact noise it was trained on, that is not robust understanding; it is memorising the noise.

Two specific, testable claims:

- **Claim B (concentrate + widen).** LAT makes each concept’s representation *cleaner* (closer to a single direction) while *also* widening the robust region around it. The interesting part is that these happen *together* — concentration and robustness are not in tension.
- **Claim C (do the proxies work?).** On real models the team used indirect measures — participation ratio, SVD concentration, SAE reconstruction quality — as stand-ins for “how much superposition is there.” With ground truth available here, we check when those proxies track real superposition and when they mislead.

~~We take as already established (from Gorton et al.) that adversarial training reduces superposition, so we do **not** re-test that — we build on it.~~

> **Revised (post-literature-critique, see `CRITIQUE.md`).** This assumption is *contested*, not settled: Bereska et al. (2025) find adversarial training's effect on superposition is **capacity-dependent** and can reverse. We therefore do **not** assume it — we test it directly with the ground-truth weight metrics (`D_i`, off-diagonal interference), which the pipeline already logs. Our own data confirms the capacity-dependence: LAT reduces interference at n/m ∈ {2,4} but not at n/m = 8.

### 3. What a good result looks like

- A clear statement of whether, in the toy setting, LAT widens the per-concept robust region relative to baseline and input-adversarial training — and whether that holds along *unseen* directions.
- A clear statement of whether concentration (concept → ~1D) and robust-region widening co-occur.
- A calibration of the proxies: which track ground-truth superposition, under which conditions, and where they break.

### 4. Model

Elhage-style toy model of superposition:

- Data: `n` features; each independently active with probability `p = 1 − S` (`S` = sparsity); active value `~ Uniform(0,1)`.
- Importance: geometric decay (uniform is a fork).
- Architecture: linear encoder `h = Wx` (`W` is `m × n`, `m < n`); decoder `x' = ReLU(Wᵀh + b)`.
- Loss: importance-weighted MSE between `x` and `x'`.
- The bottleneck `h` is “the latent.” A “concept” = one ground-truth feature.

### 5. Conditions (identical except the perturbation)

- **Baseline** — standard training.
- **Input-AT** — adversarial perturbation applied to `x`.
- **Latent-AT (LAT)** — adversarial perturbation applied to `h`: `min_θ max_{‖δ‖₂ ≤ ε} loss(x, decode(h + δ))`; `ε` scaled relative to `‖h‖`; inner loop 5–10 PGD steps; perturbation on from the start.

Hold architecture, data, seed, step count, and optimiser identical across the three.

### 6. Metrics

**Ground-truth superposition (from `W`):**

- Per-feature dimensionality `D_i = ‖W_i‖² / Σ_j (Ŵ_i · W_j)²`, where `Ŵ_i` is the unit vector of `W_i`.
- Off-diagonal interference: mean squared off-diagonal of the normalised Gram matrix `WᵀW`.

**Concept-level proxies (compute exactly as the parent project does, so they are comparable):**

- Diff-of-means per concept: `mean(h | feature i active) − mean(h | feature i inactive)`.
- Top-1 SVD explained variance and participation ratio of the per-concept activation-difference cloud.
- SAE proxies: train a TopK SAE (fixed `k`, fixed protocol across conditions) on the bottleneck; report **FVU, not raw MSE**, at fixed sparsity, plus L0 (active features) on adversarial vs clean inputs.
- Because ground truth exists, also check SAE features against true features directly (1:1 monosemanticity).

**Robust region (the core of Claim B):**

- Per concept, model frozen, find critical radius `r_i` = largest `‖δ‖₂` such that reconstruction of feature `i` stays within tolerance `τ`, measured along TWO direction sets: (1) the trained-perturbation subspace, (2) held-out random/orthogonal directions.
- Claim B is supported iff LAT increases `r_i` vs baseline/input-AT **and** the increase holds for set (2), not only set (1), **and** this co-occurs with per-concept PR dropping toward 1.

**Proxy validation (Claim C):**

- Across grid and seeds, correlate each proxy against the ground-truth superposition measures; report where they agree and diverge.

### 7. Grid and controls

- Light grid: 2–3 values of `n/m` × 2–3 values of sparsity `S`, centred on the high-sparsity regime (most analogous to a narrow safety concept). Not exhaustive.
- ≥5 seeds per cell; report mean and variance for every metric.
- Matched training across conditions (only the perturbation differs).
- Report FVU not raw MSE everywhere (avoids the activation-scale confound seen in the parent project).

### 8. Completion conditions

Done when, for every (grid cell × condition × seed):

- All Section 6 metrics are computed and logged.
- Summary artifacts produced: (a) proxy-vs-ground-truth correlation tables; (b) `r_i` basin comparison (trained vs unseen directions, across conditions); (c) per-concept PR/SVD concentration vs ground-truth superposition.
- A short written summary of whether B and C are supported, with caveats from across-seed variance.

### 9. Forks (provisional — implement the default, branch only if it fails, and log why)

- **Substrate** — default: single bottleneck. If results look unfaithful to the real “perturb early layer / measure late layer” setup, escalate to (a) a two-layer toy (perturb early, measure late), then (b) computation-in-superposition (a small net computing a function of the features).
- **Perturbation ball** — default L2. Fork: L∞.
- **Objective** — default untargeted (matches the parent Casper-style adapter). Fork: targeted (corrupt one feature) to see if it reproduces the “more diffuse” concentration seen in the targeted real-model variant.
- **ε** — default a single relative value. Fork: sweep ε.
- **Importance** — default geometric decay. Fork: uniform.
- **Grid width** — default light. Fork: widen only if B/C move with regime.

None are commitments; they are the dials to turn if the simple version misbehaves, and they can change as we go.

### 10. Deferred to later iterations (out of scope for v1)

- **Precision / quantisation bridge** (quantising `h` to ~16 levels to reproduce the fp4 smearing). Excluded for now as an additional confound and to keep v1 tractable; revisit once B/C are settled.
- **Evolvability probe** (adaptation to a new feature / compositionality) — the deeper robustness-enables-evolvability payoff; v2.
- **Anything on the real Llama models** — separate, compute-heavy track.
