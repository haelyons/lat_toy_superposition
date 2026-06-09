You are critically reviewing a completed toy-model study of latent adversarial training (LAT) in this repository, and deciding what to change. Read `SPEC.md`, `RESULTS.md`, and the code; reproduce or spot-check the key numbers before trusting them.

**Literature-review step (do this first).** Read `LITERATURE_REVIEW.md` (the deep research) and the initial research it centres on — Abbas et al. 2025, "LAT Improves the Representation of Refusal." Map our findings onto that literature and resolve the central tension explicitly: our study found LAT *widens* the robust basin (generalising to unseen directions) but does *not* concentrate concepts toward 1D, whereas Abbas et al. find LAT *concentrates* refusal, and denoising/purification theory also points to lower effective dimension. Decide whether "robust low-dimensional basin" vs "high-dimensional cloud" reframes the claims, and whether our concentration proxy (interference-dominated) measures the right thing. Engage the capacity-dependence counter-evidence (Bereska et al.); note SPEC §2 took "AT reduces superposition" as settled, which the literature contests.

**Then decide and act.** Produce a prioritised critique, implement the highest-value fixes (e.g. a cleaner concentration metric, ε sweep, targeted LAT to match Abbas, SAE-superposition direction), and log what changed and why.

Behavioural guidelines:
- Think before coding: never make silent assumptions; surface tradeoffs, ask if confused, propose multiple interpretations rather than guessing.
- Simplicity first: minimum viable code; no speculative or over-engineered single-use abstractions.
- Surgical changes: touch only what's necessary; never reformat or refactor unrelated sections.
- Goal-driven: define strict, verifiable success criteria before starting.
